import contextlib
import json
import logging
import datetime
import requests 

import emoji
import markdown2

from odoo import _, api, fields, models
from odoo.exceptions import UserError
from psycopg2 import OperationalError

_logger = logging.getLogger(__name__)





class RelatedRecordProxy:
    """
    A proxy object that provides clean access to related record fields in Jinja templates.
    Usage in templates: {{ related_record.get_field('field_name', 'default_value') }}
    When called directly, returns JSON with model name, id, and display name.
    """

    def __init__(self, record):
        self._record = record

    def get_field(self, field_name, default=""):
        """
        Get a field value from the related record.

        Args:
            field_name (str): The field name to access
            default: Default value if field doesn't exist or is empty

        Returns:
            The field value, or default if not available
        """
        if not self._record:
            return default

        try:
            if hasattr(self._record, field_name):
                value = getattr(self._record, field_name)

                # Handle different field types
                if value is None:
                    return default
                elif isinstance(value, bool):
                    return value  # Keep as boolean for Jinja
                elif hasattr(value, "name"):  # Many2one field
                    return value.name
                elif hasattr(value, "mapped"):  # Many2many/One2many field
                    return value.mapped("name")
                else:
                    return value
            else:
                _logger.debug(
                    "Field '%s' not found on record %s", field_name, self._record
                )
                return default

        except Exception as e:
            _logger.error(
                "Error getting field '%s' from record: %s", field_name, str(e)
            )
            return default

    def __getattr__(self, name):
        """Allow direct attribute access as fallback"""
        return self.get_field(name)

    def __bool__(self):
        """Return True if we have a record"""
        return bool(self._record)

    def __str__(self):
        """When called by itself, return JSON of model name, id, and display name"""
        if not self._record:
            return json.dumps({"model": None, "id": None, "display_name": None})

        return json.dumps(
            {
                "model": self._record._name,
                "id": self._record.id,
                "display_name": getattr(
                    self._record, "display_name", str(self._record)
                ),
            }
        )

    def __repr__(self):
        """Same as __str__ for consistency"""
        return self.__str__()


class LLMThread(models.Model):
    _name = "llm.thread"
    _description = "LLM Chat Thread"
    _inherit = ["mail.thread"]
    _order = "write_date DESC"

    name = fields.Char(
        string="Title",
        required=True,
    )
    user_id = fields.Many2one(
        "res.users",
        string="User",
        default=lambda self: self.env.user,
        required=True,
        ondelete="restrict",
    )
    provider_id = fields.Many2one(
        "llm.provider",
        string="Provider",
        required=True,
        ondelete="restrict",
    )
    model_id = fields.Many2one(
        "llm.model",
        string="Model",
        required=True,
        domain="[('provider_id', '=', provider_id), ('model_use', 'in', ['chat', 'multimodal'])]",
        ondelete="restrict",
    )
    active = fields.Boolean(default=True)

    # Updated fields for related record reference
    model = fields.Char(
        string="Related Document Model", help="Technical name of the related model"
    )
    res_id = fields.Many2oneReference(
        string="Related Document ID",
        model_field="model",
        help="ID of the related record",
    )



    tool_ids = fields.Many2many(
        "llm.tool",
        string="Available Tools",
        help="Tools that can be used by the LLM in this thread",
    )
    
    attachment_ids = fields.Many2many(
        'ir.attachment',
        string='All Thread Attachments',
        compute='_compute_attachment_ids',
        store=True,
        help='All attachments from all messages in this thread'
    )
    
    attachment_count = fields.Integer(
        string='Attachment Count',
        compute='_compute_attachment_count',
        store=True,
        help='Total number of attachments in this thread'
    )

    enable_expense_analysis = fields.Boolean(
        string="Enable Expense Analysis",
        default=False,
        help="Enable AI-powered expense analysis features"
    )
    
    ollama_endpoint = fields.Char(
        string="Ollama Endpoint",
        default="http://ollama:11435",
        help="Ollama API endpoint for AI analysis"
    )
    
    ollama_model = fields.Char(
        string="Ollama Model",
        default="llama3:latest",
        help="Ollama model to use for expense insights"
    )

    @api.model_create_multi
    def create(self, vals_list):
        """Set default title if not provided"""
        for vals in vals_list:
            if not vals.get("name"):
                vals["name"] = f"Chat with {self.model_id.name}"
        return super().create(vals_list)

    @api.depends('message_ids.attachment_ids')
    def _compute_attachment_ids(self):
        """Compute all attachments from all messages in this thread."""
        for thread in self:
            # Get all attachments from all messages in this thread
            all_attachments = thread.message_ids.mapped('attachment_ids')
            thread.attachment_ids = [(6, 0, all_attachments.ids)]
    
    @api.depends('attachment_ids')
    def _compute_attachment_count(self):
        """Compute the total number of attachments in this thread."""
        for thread in self:
            thread.attachment_count = len(thread.attachment_ids)

    # ============================================================================
    # MESSAGE POST OVERRIDES - Clean integration with mail.thread
    # ============================================================================

    @api.returns("mail.message", lambda value: value.id)
    def message_post(self, *, llm_role=None, message_type="comment", **kwargs):
        """Override to handle LLM-specific message types and metadata.

        Args:
            llm_role (str): The LLM role ('user', 'assistant', 'tool', 'system')
                           If provided, will automatically set the appropriate subtype
        """

        # Convert LLM role to subtype_xmlid if provided
        if llm_role:
            _, role_to_id = self.env["mail.message"].get_llm_roles()
            if llm_role in role_to_id:
                # Get the xmlid from the role
                subtype_xmlid = f"llm.mt_{llm_role}"
                kwargs["subtype_xmlid"] = subtype_xmlid

        # Handle LLM-specific subtypes and email_from generation
        if not kwargs.get("author_id") and not kwargs.get("email_from"):
            kwargs["email_from"] = self._get_llm_email_from(
                kwargs.get("subtype_xmlid"), kwargs.get("author_id"), llm_role
            )

        # Convert markdown to HTML if needed (except for tool messages which use body_json)
        if kwargs.get("body") and llm_role != "tool":
            kwargs["body"] = self._process_llm_body(kwargs["body"])

        # Create the message using standard mail.thread flow
        return super().message_post(message_type=message_type, **kwargs)

    def _get_llm_email_from(self, subtype_xmlid, author_id, llm_role=None):
        """Generate appropriate email_from for LLM messages."""
        if author_id:
            return None  # Let standard flow handle it

        provider_name = self.provider_id.name
        model_name = self.model_id.name

        if subtype_xmlid == "llm.mt_tool" or llm_role == "tool":
            return f"Tool <tool@{provider_name.lower().replace(' ', '')}.ai>"
        elif subtype_xmlid == "llm.mt_assistant" or llm_role == "assistant":
            return f"{model_name} <ai@{provider_name.lower().replace(' ', '')}.ai>"

        return None

    def _process_llm_body(self, body):
        """Process body content for LLM messages (markdown to HTML conversion)."""
        if not body:
            return body
        return markdown2.markdown(emoji.demojize(body))

    # ============================================================================
    # STREAMING MESSAGE CREATION
    # ============================================================================

    def message_post_from_stream(
        self, stream, llm_role, placeholder_text="…", **kwargs
    ):
        """Create and update a message from a streaming response.

        Args:
            stream: Generator yielding chunks of response data
            llm_role (str): The LLM role ('user', 'assistant', 'tool', 'system')
            placeholder_text (str): Text to show while streaming

        Returns:
            message: The created/updated message record
        """
        message = None
        accumulated_content = ""

        for chunk in stream:
            # Initialize message on first content
            if message is None and chunk.get("content"):
                message = self.message_post(
                    body=placeholder_text, llm_role=llm_role, author_id=False, **kwargs
                )
                yield {"type": "message_create", "message": message.message_format()[0]}

            # Handle content streaming
            if chunk.get("content"):
                accumulated_content += chunk["content"]
                message.write({"body": self._process_llm_body(accumulated_content)})
                yield {"type": "message_chunk", "message": message.message_format()[0]}

            # Handle errors
            if chunk.get("error"):
                yield {"type": "error", "error": chunk["error"]}
                return message

        # Final update for assistant message
        if message and accumulated_content:
            message.write({"body": self._process_llm_body(accumulated_content)})
            yield {"type": "message_update", "message": message.message_format()[0]}

        return message

    # ============================================================================
    # GENERATION FLOW - Refactored to use message_post with roles
    # ============================================================================

    def generate(self, user_message_body, **kwargs):
        """Main generation method with PostgreSQL advisory locking."""
        self.ensure_one()

        with self._generation_lock():
            last_message = False

            # 1) Always post the user message
            if user_message_body:
                last_message = self.message_post(
                    body=user_message_body,
                    llm_role="user",
                    author_id=self.env.user.partner_id.id,
                    **kwargs,
                )
                yield {"type": "message_create", "message": last_message.message_format()[0]}

            # 2) Expense path → delegate to generate_expense_response
            if self.enable_expense_analysis and user_message_body and self._is_expense_query(user_message_body):
                resp = self.generate_expense_response(user_message_body)
                assistant_message = self.message_post(
                    body=resp,
                    llm_role="assistant",
                    author_id=False,
                    **kwargs,
                )
                yield {"type": "message_create", "message": assistant_message.message_format()[0]}
                return assistant_message  # IMPORTANT: prevents generic path from overriding

            # 3) Generic assistant path
            last_message = yield from self.generate_messages(last_message)
            return last_message



    def generate_messages(self, last_message=None):
        """Generate messages - to be overridden by llm_assistant module."""
        raise UserError(
            _("Please install the llm_assistant module for actual AI generation.")
        )

    def get_context(self, base_context=None):
        context = {
            **(base_context or {}),
            "thread_id": self.id,
        }

        try:
            related_record = self.env[self.model].browse(self.res_id)
            if related_record:
                context["related_record"] = RelatedRecordProxy(related_record)
                context["related_model"] = self.model
                context["related_res_id"] = self.res_id
            else:
                context["related_record"] = None
                context["related_model"] = None
                context["related_res_id"] = None
        except Exception as e:
            _logger.warning(
                "Error accessing related record %s,%s: %s", self.model, self.res_id, e
            )

        ##Expense Analysis
        # Fixed expense analysis context - removed non-existent methods
        if self.enable_expense_analysis:
            context.update({
                'expense_analyzer': {
                    'analyze_by_product': self.analyze_expenses_by_product,
                    'analyze_trends': self.analyze_expense_trends,
                    'generate_insights': self.generate_expense_insights,
                    'get_summary': self.get_expense_summary,
                    'get_pending': self.get_pending_expenses,
                    
                }
            })

        return context
    
    # ============================================================================
    # EXPENSE ANALYSIS METHODS - Custom Finance AI Integration
    # ============================================================================

    # Fixed expense analysis methods for LLM thread

    def _is_expense_query(self, message):
        expense_keywords = [
            'expense', 'expenses', 'spending', 'spend', 'cost', 'budget',
            'financial', 'money', 'expense analysis', 'analyze expenses',
            'spend analysis', 'by product', 'insights', 'expenditure',
            'breakdown', 'pending', 'approval', 'reimburse'
        ]
        m = (message or "").lower()
        return any(k in m for k in expense_keywords)

    def _extract_timeframe(self, message):
        m = (message or "").lower()
        if 'this quarter' in m or 'current quarter' in m:
            return 'this_quarter'
        elif 'this month' in m or 'current month' in m:
            return 'this_month'
        elif 'last month' in m or 'previous month' in m:
            return 'last_month'
        elif 'this year' in m or 'current year' in m:
            return 'this_year'
        elif 'last 30 days' in m or '30 days' in m:
            return 'last_30_days'
        return 'this_month'

    def _parse_month_year(self, text):
        try:
            import calendar, re, datetime
            m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', (text or '').lower())
            if not m:
                return None, None
            month = list(calendar.month_name).index(m.group(1).capitalize())
            year = int(m.group(2))
            start = datetime.date(year, month, 1)
            last_day = calendar.monthrange(year, month)[11]
            end = datetime.date(year, month, last_day)
            return start, end
        except Exception as e:
            _logger.warning("Error parsing month/year: %s", e)
            return None, None

    def generate_expense_response(self, message, **kwargs):
        self.ensure_one()
        m = (message or "").lower()
        try:
            # No trend route anymore
            if 'pending' in m or 'approval' in m or 'reimburse' in m:
                resp = self.get_pending_expenses()
            elif 'summary' in m or 'overview' in m:
                resp = self.get_expense_summary()
            elif 'insight' in m or 'recommendation' in m or 'ai' in m:
                resp = self.generate_expense_insights(message)
            else:
                # Default: by-product with month-year support or fixed timeframe
                start, end = self._parse_month_year(message)
                if start and end:
                    resp = self.analyze_expenses_by_product(date_from=start, date_to=end)
                else:
                    timeframe = self._extract_timeframe(message)
                    resp = self.analyze_expenses_by_product(timeframe)
            if not resp or resp.strip() == "":
                resp = "No expense data found or unable to generate a response."
            return resp
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("LLMThread(%s): error in generate_expense_response: %s", self.id, e)
            return f"❌ Error generating expense response: {str(e)}"



    def _extract_product_name(self, message):
        """Extract product name from message."""
        try:
            # Split message into tokens and look for capitalized words
            tokens = [w.strip(",.!?:;") for w in message.split()]
            candidates = [w for w in tokens if w.istitle() and len(w) > 2]
            
            if candidates:
                # Join consecutive capitalized words (e.g., "Laptop Computer")
                product_name = ' '.join(candidates[:2])  # Take first 2 words max
                return product_name
            
            # Fallback: search for products in database
            ProductTmpl = self.env['product.template']
            for word in tokens:
                if len(word) > 3:  # Only check meaningful words
                    rec = ProductTmpl.search([('name', 'ilike', word)], limit=1)
                    if rec:
                        return rec.name
            
            return None
        except Exception as e:
            _logger.warning(f"Error extracting product name: {e}")
            return None

    def analyze_expenses_by_product(self, timeframe='this_month', date_from=None, date_to=None):
        try:
            today = datetime.date.today()
            if date_from and date_to:
                start, end = date_from, date_to
            else:
                end = today
                if timeframe == 'this_quarter':
                    month = (today.month - 1) // 3 * 3 + 1
                    start = datetime.date(today.year, month, 1)
                elif timeframe == 'this_year':
                    start = datetime.date(today.year, 1, 1)
                elif timeframe == 'last_month':
                    import calendar
                    if today.month == 1:
                        # last month is December of previous year
                        start = datetime.date(today.year - 1, 12, 1)
                        end = datetime.date(today.year - 1, 12, calendar.monthrange(today.year - 1, 12)[1])
                    else:
                        start = datetime.date(today.year, today.month - 1, 1)
                        end = datetime.date(today.year, today.month - 1, calendar.monthrange(today.year, today.month - 1)[1])
                elif timeframe == 'last_30_days':
                    start = end - datetime.timedelta(days=30)
                else:
                    start = end.replace(day=1)

            domain = [
                ('state', 'in', ['approved']),
                ('date', '>=', start),
                ('date', '<=', end),
                ('product_id', '!=', False),
            ]
            
            expenses = self.env['hr.expense'].search(domain, order='date desc')
            
            if not expenses:
                return f'<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4>📊 No Expense Data Found</h4><p>No approved expenses found for the period {start.strftime("%b %d, %Y")} to {end.strftime("%b %d, %Y")}</p></div>'
            
            # Group by product for summary table
            product_totals = {}
            expense_details = []
            
            total_amount = 0.0
            for e in expenses:
                emp = e.employee_id.name or 'Unknown Employee'
                dt = e.date.strftime("%Y-%m-%d") if e.date else "No Date"
                amt = e.total_amount or 0.0
                product_name = e.product_id.product_tmpl_id.name or e.product_id.display_name or e.name or 'Other Expenses'
                
                total_amount += amt
                
                # Group by product for summary
                if product_name in product_totals:
                    product_totals[product_name]['amount'] += amt
                    product_totals[product_name]['count'] += 1
                else:
                    product_totals[product_name] = {'amount': amt, 'count': 1}
                
                # Store individual expense details
                expense_details.append({
                    'name': e.name,
                    'employee': emp,
                    'date': dt,
                    'amount': amt,
                    'product': product_name,
                    'state': e.state
                })
            
            # Build HTML output
            html_lines = []
            
            # Header with period info
            html_lines.append('<div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;">')
            html_lines.append('<h2 style="margin: 0 0 5px 0; font-size: 18px;">📊 EXPENSE ANALYSIS BY PRODUCT</h2>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">📅 Period: {start.strftime("%b %d, %Y")} to {end.strftime("%b %d, %Y")}</p>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">💰 Total Analyzed: ₹{total_amount:,.2f} ({len(expenses)} expenses)</p>')
            html_lines.append('</div>')
            
            # Product summary table
            if product_totals:
                html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
                html_lines.append('<h4 style="color: white; background: #28a745; padding: 8px; margin: 0; font-size: 14px;">📈 EXPENSES BY PRODUCT/SERVICE</h4>')
                
                html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
                html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
                html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">Product/Service</th>')
                html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Amount</th>')
                html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
                html_lines.append('</tr></thead><tbody>')
                
                # Sort by amount descending
                sorted_products = sorted(product_totals.items(), key=lambda x: x[1]['amount'], reverse=True)
                
                for i, (product, data) in enumerate(sorted_products):
                    pct = (data['amount'] / total_amount) * 100 if total_amount > 0 else 0
                    name = (product[:30] + '...') if len(product) > 30 else product
                    row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                    
                    html_lines.append(f'<tr style="background-color: {row_bg};">')
                    html_lines.append(f'<td style="padding: 6px; color: #333;" title="{product}">{name}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: #28a745;">₹{data["amount"]:,.2f}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; color: #28a745; font-weight: bold;">{pct:.1f}%</td>')
                    html_lines.append('</tr>')
                
                # Total row
                html_lines.append('<tr style="background: #e8f5e8; font-weight: bold; border-top: 2px solid #28a745;">')
                html_lines.append('<td style="padding: 8px; color: #155724;">TOTAL</td>')
                html_lines.append(f'<td style="padding: 8px; text-align: right; color: #155724;">₹{total_amount:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: right; color: #155724;">100.0%</td>')
                html_lines.append('</tr>')
                
                html_lines.append('</tbody></table></div>')
            
            # Detailed expense list table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6c757d; padding: 8px; margin: 0; font-size: 14px;">📋 DETAILED EXPENSE BREAKDOWN</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Description</th>')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Employee</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Date</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Amount</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Status</th>')
            html_lines.append('</tr></thead><tbody>')
            
            # Show top 15 individual expenses
            for i, detail in enumerate(expense_details[:15]):
                desc = (detail['name'][:25] + '...') if len(detail['name']) > 25 else detail['name']
                emp = (detail['employee'][:15] + '...') if len(detail['employee']) > 15 else detail['employee']
                row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                
                html_lines.append(f'<tr style="background-color: {row_bg};">')
                html_lines.append(f'<td style="padding: 5px; color: #333;" title="{detail["name"]}">{desc}</td>')
                html_lines.append(f'<td style="padding: 5px; color: #333;" title="{detail["employee"]}">{emp}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666;">{detail["date"]}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; font-weight: bold; color: #28a745;">₹{detail["amount"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #17a2b8; font-weight: bold;">{detail["state"]}</td>')
                html_lines.append('</tr>')
            
            if len(expense_details) > 15:
                remaining_count = len(expense_details) - 15
                remaining_amount = sum(d['amount'] for d in expense_details[15:])
                html_lines.append('<tr style="background: #fff3cd; font-style: italic;">')
                html_lines.append(f'<td colspan="4" style="padding: 8px; color: #856404;">... and {remaining_count} more expenses totaling ₹{remaining_amount:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: center; color: #856404;">approved</td>')
                html_lines.append('</tr>')
            
            # Final total row
            html_lines.append('<tr style="background: #e8f5e8; font-weight: bold; border-top: 2px solid #28a745;">')
            html_lines.append('<td colspan="3" style="padding: 8px; color: #155724;">TOTAL ANALYZED</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #155724;">₹{total_amount:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: center; color: #155724;">-</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error in analyze_expenses_by_product: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4>❌ Error</h4><p>Error analyzing expenses: {str(e)}</p></div>'



    def get_pending_expenses(self):
        try:
            pending_states = ['reported']  # pending == reported (to be reimbursed)
            expenses = self.env['hr.expense'].search(
                [('state', 'in', pending_states)],
                order='date desc', limit=50
            )
            if not expenses:
                return '<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4>✅ No Pending Expenses</h4><p>All expenses are processed!</p></div>'
            
            # Group by product for summary table
            product_totals = {}
            employee_details = []
            
            total_pending = 0.0
            for e in expenses:
                emp = e.employee_id.name or 'Unknown Employee'
                dt = e.date.strftime("%Y-%m-%d") if e.date else "No Date"
                amt = e.total_amount or 0.0
                product_name = e.product_id.name or e.name or 'Other Expenses'
                
                total_pending += amt
                
                # Group by product for summary
                if product_name in product_totals:
                    product_totals[product_name] += amt
                else:
                    product_totals[product_name] = amt
                
                # Store individual expense details
                employee_details.append({
                    'name': e.name,
                    'employee': emp,
                    'date': dt,
                    'amount': amt,
                    'product': product_name,
                    'state': e.state
                })
            
            # Build HTML output matching the expense insights format
            html_lines = []
            
            # Header
            html_lines.append('<div style="background: linear-gradient(135deg, #dc3545, #fd7e14); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;">')
            html_lines.append('<h2 style="margin: 0 0 5px 0; font-size: 18px;">⏳ PENDING EXPENSE REIMBURSEMENTS</h2>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">💰 Total Pending: ₹{total_pending:,.2f} ({len(expenses)} expenses)</p>')
            html_lines.append('</div>')
            
            # Product summary table (using same format as expense insights)
            if product_totals:
                html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
                html_lines.append('<h4 style="color: white; background: #dc3545; padding: 8px; margin: 0; font-size: 14px;">📊 PENDING EXPENSES BY PRODUCT/SERVICE</h4>')
                
                html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
                html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
                html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">Product/Service</th>')
                html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Amount</th>')
                html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
                html_lines.append('</tr></thead><tbody>')
                
                # Sort by amount descending
                sorted_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)
                
                for i, (product, amt) in enumerate(sorted_products[:8]):  # Show top 8
                    pct = (amt / total_pending) * 100 if total_pending > 0 else 0
                    name = (product[:30] + '...') if len(product) > 30 else product
                    row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                    
                    html_lines.append(f'<tr style="background-color: {row_bg};">')
                    html_lines.append(f'<td style="padding: 6px; color: #333;" title="{product}">{name}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: #dc3545;">₹{amt:,.2f}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; color: #dc3545; font-weight: bold;">{pct:.1f}%</td>')
                    html_lines.append('</tr>')
                
                # Total row
                html_lines.append('<tr style="background: #ffeaa7; font-weight: bold; border-top: 2px solid #fdcb6e;">')
                html_lines.append('<td style="padding: 8px; color: #e17055;">TOTAL</td>')
                html_lines.append(f'<td style="padding: 8px; text-align: right; color: #e17055;">₹{total_pending:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: right; color: #e17055;">100.0%</td>')
                html_lines.append('</tr>')
                
                html_lines.append('</tbody></table></div>')
            
            # Detailed expense list table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6c757d; padding: 8px; margin: 0; font-size: 14px;">📋 DETAILED PENDING EXPENSES</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Description</th>')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Employee</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Date</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Amount</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Status</th>')
            html_lines.append('</tr></thead><tbody>')
            
            # Show top 15 individual expenses
            for i, detail in enumerate(employee_details[:15]):
                desc = (detail['name'][:25] + '...') if len(detail['name']) > 25 else detail['name']
                emp = (detail['employee'][:15] + '...') if len(detail['employee']) > 15 else detail['employee']
                row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                
                html_lines.append(f'<tr style="background-color: {row_bg};">')
                html_lines.append(f'<td style="padding: 5px; color: #333;" title="{detail["name"]}">{desc}</td>')
                html_lines.append(f'<td style="padding: 5px; color: #333;" title="{detail["employee"]}">{emp}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666;">{detail["date"]}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; font-weight: bold; color: #dc3545;">₹{detail["amount"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #fd7e14; font-weight: bold;">{detail["state"]}</td>')
                html_lines.append('</tr>')
            
            if len(employee_details) > 15:
                remaining_count = len(employee_details) - 15
                remaining_amount = sum(d['amount'] for d in employee_details[15:])
                html_lines.append('<tr style="background: #fff3cd; font-style: italic;">')
                html_lines.append(f'<td colspan="4" style="padding: 8px; color: #856404;">... and {remaining_count} more expenses totaling ₹{remaining_amount:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: center; color: #856404;">pending</td>')
                html_lines.append('</tr>')
            
            # Final total row
            html_lines.append('<tr style="background: #ffeaa7; font-weight: bold; border-top: 2px solid #fdcb6e;">')
            html_lines.append('<td colspan="3" style="padding: 8px; color: #e17055;">TOTAL PENDING</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #e17055;">₹{total_pending:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: center; color: #e17055;">-</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error getting pending expenses: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4>❌ Error</h4><p>{str(e)}</p></div>'

    
    
    
    def _ascii_bars(self, totals):
        """HTML bar chart with properly colored and sized progress bars"""
        if not totals:
            return '<div style="text-align: center; color: #666; padding: 20px;">No data available</div>'
        
        total_amt = sum(a for _, a in totals) or 1.0
        html_lines = ['<div style="font-family: Arial, sans-serif; line-height: 1.6; margin: 10px 0; background: #f8f9fa; padding: 15px; border-radius: 6px;">']
        max_bar_width = 300  # Fixed container width
    
        for label, amt in totals:
            pct = amt / total_amt
            # Calculate proportional width - this is key!
            bar_width = max(int(pct * max_bar_width), 10)  # Minimum 10px for visibility
            label_disp = label if len(label) <= 25 else label[:22] + "..."
            
            html_lines.append('<div style="display: flex; align-items: center; margin: 8px 0; padding: 3px 0;">')
            
            # Label with fixed width
            html_lines.append(f'<div title="{label}" style="width: 180px; font-weight: bold; color: #333; margin-right: 12px; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label_disp}</div>')
            
            # Fixed container (gray track) - SAME SIZE FOR ALL
            html_lines.append(f'<div style="width: {max_bar_width}px; height: 20px; background-color: #e8e8e8; border-radius: 10px; margin-right: 12px; border: 1px solid #ccc; position: relative; overflow: hidden;">')
            
            # Variable colored bar (green fill) - SIZE VARIES BY PERCENTAGE
            html_lines.append(f'<div style="height: 100%; width: {bar_width}px; background: linear-gradient(90deg, #4caf50, #66bb6a); border-radius: 9px; box-shadow: inset 0 1px 2px rgba(255,255,255,0.3); transition: width 0.3s ease;"></div>')
            
            html_lines.append('</div>') # Close bar container
            
            # Percentage with better styling
            html_lines.append(f'<div style="width: 60px; text-align: right; font-weight: bold; color: #2e7d32; font-size: 13px;">{pct*100:.1f}%</div>')
            
            html_lines.append('</div>') # Close row container
        
        html_lines.append('</div>')
        return '\n'.join(html_lines)

    def _totals_by_product(self, state_list, start, end):
        """No changes - keeping original logic"""
        exps = self.env['hr.expense'].search([
            ('state','in', state_list),
            ('date','>=', start), ('date','<=', end),
            ('product_id','!=', False),
        ])
        totals = {}
        for e in exps:
            pname = e.product_id.product_tmpl_id.name or e.product_id.display_name or e.name or 'Unknown'
            totals[pname] = totals.get(pname, 0.0) + (e.total_amount or 0.0)
        return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    def _format_currency(self, amt):
        """No changes - keeping original logic"""
        return f"₹{amt:,.2f}"

    def _simple_table(self, totals, title):
        """Compact HTML table"""
        if not totals:
            return f'<div style="margin: 10px 0; padding: 10px; background: #fff3e0; border-radius: 6px; color: #666;"><strong>{title}</strong><br>No data available</div>'
        
        total_amt = sum(a for _, a in totals) or 1.0
        html_lines = []
        html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
        html_lines.append(f'<h4 style="color: white; background: #1976d2; padding: 8px; margin: 0; font-size: 14px;">{title}</h4>')
        
        html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
        html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
        html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">Product/Service</th>')
        html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Amount</th>')
        html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
        html_lines.append('</tr></thead><tbody>')
        
        for i, (label, amt) in enumerate(totals[:8]):  # Show only top 8
            pct = (amt / total_amt) * 100
            name = (label[:30] + '...') if len(label) > 30 else label
            row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
            
            html_lines.append(f'<tr style="background-color: {row_bg};">')
            html_lines.append(f'<td style="padding: 6px; color: #333;" title="{label}">{name}</td>')
            html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: #2e7d32;">{self._format_currency(amt)}</td>')
            html_lines.append(f'<td style="padding: 6px; text-align: right; color: #1976d2; font-weight: bold;">{pct:.1f}%</td>')
            html_lines.append('</tr>')
        
        # Compact total row
        html_lines.append('<tr style="background: #e8f5e9; font-weight: bold; border-top: 2px solid #4caf50;">')
        html_lines.append('<td style="padding: 8px; color: #2e7d32;">TOTAL</td>')
        html_lines.append(f'<td style="padding: 8px; text-align: right; color: #2e7d32;">{self._format_currency(total_amt)}</td>')
        html_lines.append('<td style="padding: 8px; text-align: right; color: #2e7d32;">100.0%</td>')
        html_lines.append('</tr>')
        
        html_lines.append('</tbody></table></div>')
        return '\n'.join(html_lines)

    def _summary_stats(self, approved_totals, reported_totals, all_totals):
        """Compact summary statistics"""
        ap = sum(a for _, a in approved_totals) if approved_totals else 0.0
        rp = sum(a for _, a in reported_totals) if reported_totals else 0.0
        al = sum(a for _, a in all_totals) if all_totals else 0.0
        top_name, top_amt = ("None", 0.0)
        if all_totals:
            top_name, top_amt = all_totals[0]
        
        html_lines = []
        html_lines.append('<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #9c27b0;">')
        html_lines.append('<h3 style="color: #6a1b9a; margin: 0 0 10px 0; font-size: 16px; text-align: center;">SUMMARY STATISTICS</h3>')
        
        # Horizontal stats layout
        html_lines.append('<div style="display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap;">')
        
        # Three stat boxes in a row
        stats = [
            ("✅", "Approved", ap, "#4caf50"),
            ("📋", "Reported", rp, "#ff9800"), 
            ("📈", "All States", al, "#2196f3")
        ]
        
        for icon, label, amount, color in stats:
            html_lines.append(f'<div style="background: white; padding: 10px; border-radius: 6px; flex: 1; min-width: 120px; border-left: 3px solid {color};">')
            html_lines.append(f'<div style="font-size: 12px; color: {color}; font-weight: bold;">{icon} {label}</div>')
            html_lines.append(f'<div style="color: {color}; font-size: 16px; font-weight: bold;">{self._format_currency(amount)}</div>')
            html_lines.append('</div>')
        
        html_lines.append('</div>')
        
        # Top expense in a compact format
        if al > 0 and top_amt > 0:
            share = (top_amt / al) * 100
            html_lines.append('<div style="background: white; padding: 10px; border-radius: 6px; border-left: 3px solid #e91e63;">')
            html_lines.append(f'<div style="font-size: 12px; color: #c2185b; font-weight: bold;">🔝 Top Category: {top_name}</div>')
            html_lines.append(f'<div style="font-size: 14px; color: #c2185b; font-weight: bold;">{self._format_currency(top_amt)} ({share:.1f}%)</div>')
            html_lines.append('</div>')
        
        html_lines.append('</div>')
        return '\n'.join(html_lines)

    def _visual_chart(self, totals, title):
        """Compact visual chart with proper header"""
        html_lines = []
        html_lines.append('<div style="background: #1976d2; color: white; padding: 8px; border-radius: 6px 6px 0 0; margin: 15px 0 0 0;">')
        html_lines.append(f'<h4 style="margin: 0; font-size: 14px; text-align: center;">{title}</h4>')
        html_lines.append('</div>')
        
        if not totals:
            html_lines.append('<div style="background: white; padding: 15px; text-align: center; color: #666; border-radius: 0 0 6px 6px; border: 1px solid #ddd; border-top: none;">No data available</div>')
            return '\n'.join(html_lines)
        
        html_lines.append('<div style="background: white; border-radius: 0 0 6px 6px; border: 1px solid #ddd; border-top: none;">')
        html_lines.append(self._ascii_bars(totals))
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)

    def generate_expense_insights(self, prompt):
        """Fixed expense insights with proper header and bar colors"""
        try:
            import datetime
            # Resolve period (month-year preferred if present)
            start, end = self._parse_month_year(prompt)
            if not (start and end):
                today = datetime.date.today()
                month = (today.month - 1) // 3 * 3 + 1
                start = datetime.date(today.year, month, 1)
                end = today
            
            # Build three visualizations by product
            approved_totals = self._totals_by_product(['approved'], start, end)
            reported_totals = self._totals_by_product(['reported'], start, end)
            all_totals = self._totals_by_product(['approved','reported','to_submit'], start, end)
            
            # Fixed header with proper styling
            period_header = '<div style="background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; border-radius: 8px; margin: 10px 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
            period_header += '<h2 style="margin: 0 0 8px 0; font-size: 22px; font-weight: bold;">EXPENSE ANALYSIS REPORT</h2>'
            period_header += f'<p style="margin: 0; font-size: 14px; opacity: 0.9; font-weight: 300;">{start.strftime("%B %d, %Y")} to {end.strftime("%B %d, %Y")}</p>'
            period_header += '</div>'
            
            # Generate all sections with reduced spacing
            summary_stats = self._summary_stats(approved_totals, reported_totals, all_totals)
            approved_table = self._simple_table(approved_totals, "✅ APPROVED EXPENSES")
            approved_chart = self._visual_chart(approved_totals, "APPROVED BREAKDOWN")
            all_table = self._simple_table(all_totals, "📊 ALL EXPENSES")
            all_chart = self._visual_chart(all_totals, "ALL EXPENSES BREAKDOWN")
            
            # Include breakdown text from canonical function
            try:
                breakdown = self.analyze_expenses_by_product(date_from=start, date_to=end)
            except:
                breakdown = "Detailed breakdown unavailable"
            
            # Combine with minimal spacing
            viz = period_header + summary_stats + approved_table + approved_chart + all_table + all_chart
            
            # Compact AI insights
            try:
                ai_prompt = f"""Based on expense data, provide 3 brief insights:
                - Approved: ₹{sum(a for _, a in approved_totals) if approved_totals else 0:,.2f}
                - Total: ₹{sum(a for _, a in all_totals) if all_totals else 0:,.2f}
                - Top: {'; '.join([f"{name}: ₹{amt:,.2f}" for name, amt in all_totals[:3]]) if all_totals else 'None'}
                User: {prompt}"""
                
                import requests
                resp = requests.post(
                    f'{self.ollama_endpoint}/api/generate',
                    json={
                        'model': self.ollama_model, 
                        'prompt': ai_prompt, 
                        'stream': False,
                        'options': {'temperature': 0.5, 'max_tokens': 300}
                    },
                    timeout=30
                )
                
                if resp.status_code == 200:
                    ai_response = resp.json().get('response', '')
                    ai_insights = '<div style="background: #8e24aa; color: white; padding: 12px; border-radius: 8px; margin: 15px 0;">'
                    ai_insights += '<h4 style="margin: 0 0 8px 0; font-size: 14px;">🤖 AI INSIGHTS</h4>'
                    ai_insights += f'<div style="font-size: 13px; line-height: 1.4;">{ai_response}</div></div>'
                    return viz + ai_insights
                else:
                    return viz + '<div style="background: #fff3cd; padding: 8px; border-radius: 6px; color: #856404; margin: 10px 0; font-size: 12px;">⚠ AI analysis unavailable</div>'
                    
            except ImportError:
                return viz + '<div style="background: #fff3cd; padding: 8px; border-radius: 6px; color: #856404; margin: 10px 0; font-size: 12px;">⚠ AI analysis unavailable (requests library missing)</div>'
            except Exception as e:
                return viz + '<div style="background: #f8d7da; padding: 8px; border-radius: 6px; color: #721c24; margin: 10px 0; font-size: 12px;">❌ AI analysis error</div>'
                
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error generating expense insights: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4>❌ Error</h4><p>{str(e)}</p></div>'

    # POSTGRESQL ADVISORY LOCK IMPLEMENTATION
    

    def _acquire_thread_lock(self):
        """Acquire PostgreSQL advisory lock for this thread."""
        self.ensure_one()

        try:
            query = "SELECT pg_try_advisory_lock(%s)"
            self.env.cr.execute(query, (self.id,))
            result = self.env.cr.fetchone()

            if not result or not result[0]:
                raise UserError(
                    _("Thread is currently generating a response. Please wait.")
                )

            _logger.info(f"Acquired advisory lock for thread {self.id}")

        except UserError:
            raise
        except OperationalError as e:
            _logger.error(f"Database error acquiring lock for thread {self.id}: {e}")
            raise UserError(_("Database error acquiring thread lock.")) from e
        except Exception as e:
            _logger.error(f"Unexpected error acquiring lock for thread {self.id}: {e}")
            raise UserError(_("Failed to acquire thread lock.")) from e

    def _release_thread_lock(self):
        """Release PostgreSQL advisory lock for this thread."""
        self.ensure_one()

        try:
            query = "SELECT pg_advisory_unlock(%s)"
            self.env.cr.execute(query, (self.id,))
            result = self.env.cr.fetchone()

            success = result and result[0]
            if success:
                _logger.info(f"Released advisory lock for thread {self.id}")
            else:
                _logger.warning(f"Advisory lock for thread {self.id} was not held")

            return success

        except Exception as e:
            _logger.error(f"Error releasing lock for thread {self.id}: {e}")
            return False

    @contextlib.contextmanager
    def _generation_lock(self):
        """Context manager for thread generation with automatic lock cleanup."""
        self.ensure_one()

        self._acquire_thread_lock()

        try:
            _logger.info(f"Starting locked generation for thread {self.id}")
            yield self

        finally:
            released = self._release_thread_lock()
            if released:
                _logger.info(f"Finished locked generation for thread {self.id}")
            else:
                _logger.warning(f"Lock release failed for thread {self.id}")
  


    # ============================================================================
    # ODOO HOOKS AND CLEANUP
    # ============================================================================

    @api.ondelete(at_uninstall=False)
    def _unlink_llm_thread(self):
        unlink_ids = [record.id for record in self]
        self.env["bus.bus"]._sendone(
            self.env.user.partner_id, "llm.thread/delete", {"ids": unlink_ids}
        )
