import contextlib
import json
import logging
import datetime
import requests 
import math
import statistics
import random
from datetime import date , timedelta

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

    enable_inventory_analysis = fields.Boolean(
        string="Enable Inventory Analysis",
        default=False,
        help="Enable AI-powered inventory analysis features"
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
                return assistant_message

            # 3) Inventory path → delegate to generate_inventory_response
            if self.enable_inventory_analysis and user_message_body and self._is_inventory_query(user_message_body):
                resp = self.generate_inventory_response(user_message_body)
                assistant_message = self.message_post(
                    body=resp,
                    llm_role="assistant",
                    author_id=False,
                    **kwargs,
                )
                yield {"type": "message_create", "message": assistant_message.message_format()[0]}
                return assistant_message

            # 4) Generic assistant path
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

        # Expense Analysis
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

        # Inventory Analysis
        if self.enable_inventory_analysis:
            context.update({
                'inventory_analyzer': {
                    'reorder_suggestions': self.intelligent_reorder_suggestions,
                    'abc_analysis': self.generate_abc_analysis,
                    'inventory_turnover': self.calculate_inventory_turnover
                }
            })

        return context

    
    # ============================================================================
    # EXPENSE ANALYSIS METHODS - Custom Finance AI Integration
    # ============================================================================

  
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
    

   
    

    # ============================================================================
    # POSTGRESQL ADVISORY LOCK IMPLEMENTATION
    # ============================================================================


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

    
    # ============================================================================
    # Inventory
    # ============================================================================
    def _is_inventory_query(self, message):
        """Enhanced inventory query detection with more keywords"""
        inventory_keywords = [
            'inventory', 'stock', 'reorder', 'restock', 'replenish', 
            'stock level', 'inventory level', 'low stock', 'out of stock',
            'purchase suggestion', 'order recommendation', 'intelligent reorder',
            'smart reorder', 'inventory analysis', 'stock analysis',
            'warehouse', 'product stock', 'stockout',
            'safety stock', 'economic order quantity', 'eoq', 'lead time',
            'abc analysis', 'inventory turnover', 'carrying cost', 'shortage'
        ]
        m = (message or "").lower()
        return any(k in m for k in inventory_keywords)

    def generate_inventory_response(self, message, **kwargs):
        """Enhanced inventory analysis response with realistic data processing"""
        self.ensure_one()
        m = (message or "").lower().strip()
        
        try:
            # Check if we have any products first
            product_count = self.env['product.product'].search_count([
                ('type', '=', 'product'),
                ('active', '=', True)
            ])
            
            if product_count == 0:
                return '''
                <div class="alert alert-warning">
                    <strong>No Products Found:</strong> No products are available in the system for inventory analysis.
                    Please add products first to use inventory management features.
                </div>
                '''
            
            # Enhanced keyword matching with better logic
            if any(keyword in m for keyword in ['reorder', 'restock', 'replenish', 'low stock', 'stock alert']):
                resp = self.intelligent_reorder_suggestions()
            
            elif any(keyword in m for keyword in ['abc analysis', 'abc']):
                resp = self.generate_abc_analysis()
            elif any(keyword in m for keyword in ['turnover', 'inventory turnover']):
                resp = self.calculate_inventory_turnover()
            elif any(keyword in m for keyword in ['stock level', 'inventory level', 'stock analysis', 'inventory report', 'stock report']):
                resp = self.analyze_stock_levels()
            elif any(keyword in m for keyword in ['help', 'commands', 'what can you do']):
                resp = self._generate_help_response()
            else:
                # Intelligent default based on current inventory state
                critical_items = self._get_critical_stock_count()
                if critical_items > 0:
                    resp = f"""
                    <div class="alert alert-danger" style="margin-bottom: 15px;">
                        <strong> Alert:</strong> {critical_items} products have critical stock levels that need immediate attention.
                    </div>
                    {self.intelligent_reorder_suggestions()}
                    """
                else:
                    resp = f"""
                    <div class="alert alert-info" style="margin-bottom: 15px;">
                        <strong> Status:</strong> No critical stock alerts. Running comprehensive inventory analysis.
                    </div>
                    {self.analyze_stock_levels()}
                    """
            
            if not resp or resp.strip() == "":
                resp = '''
                <div class="alert alert-warning">
                    <strong>Analysis Incomplete:</strong> Unable to generate inventory analysis. 
                    This may be due to insufficient data or system limitations.
                </div>
                '''
            
            return resp
        
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("LLMThread(%s): error in generate_inventory_response: %s", self.id, e)
            return f'''
            <div class="alert alert-danger">
                <strong>❌ System Error:</strong> {str(e)[:100]}...
                <br><small>Please contact the system administrator or try again later.</small>
            </div>
            '''

    def _get_critical_stock_count(self):
        """Get count of products with critical stock levels"""
        try:
            # Products with zero or very low stock
            critical_count = self.env['product.product'].search_count([
                ('type', '=', 'product'),
                ('active', '=', True),
                ('qty_available', '<=', 5)  # Consider ≤5 as critical
            ])
            return critical_count
        except Exception:
            return 0



    def _calculate_enhanced_features(self, product):
        """Calculate enhanced features with more realistic data"""
        try:
            features = {
                'seasonal_factor': self._get_realistic_seasonal_factor(product.id),
                'lead_time_variability': self._calculate_realistic_lead_time_variance(product.id),
                'demand_volatility': self._calculate_realistic_demand_volatility(product.id),
                'supplier_reliability': self._assess_realistic_supplier_reliability(product.id),
                'price_trend': self._analyze_realistic_price_trends(product.id),
                'abc_classification': self._get_abc_classification(product.id),
                'stock_coverage_days': self._calculate_stock_coverage_days(product.id)
            }
            return features
        except Exception:
            # Return default safe values if calculation fails
            return {
                'seasonal_factor': 1.0,
                'lead_time_variability': 0.3,
                'demand_volatility': 0.4,
                'supplier_reliability': 0.8,
                'price_trend': 0.0,
                'abc_classification': 'B',
                'stock_coverage_days': 30
            }

    def _calculate_reorder_metrics(self, product, features):
        """Calculate realistic reorder metrics based on actual business logic"""
        try:
            current_stock = product.qty_available
            
            # Base calculations using realistic business rules
            avg_daily_demand = self._calculate_avg_daily_demand(product.id)
            lead_time_days = self._get_realistic_lead_time(product.id)
            
            # Safety stock calculation based on service level (95%)
            service_level = 0.95
            z_score = 1.65  # 95% service level
            demand_during_lead_time = avg_daily_demand * lead_time_days
            
            # Safety stock considers demand variability and lead time uncertainty
            safety_stock = z_score * math.sqrt(
                (lead_time_days * features['demand_volatility'] * avg_daily_demand) +
                (avg_daily_demand ** 2 * features['lead_time_variability'])
            )
            
            # Reorder point = demand during lead time + safety stock
            optimal_reorder = max(1, int(demand_during_lead_time + safety_stock))
            
            # Economic Order Quantity (EOQ) consideration
            carrying_cost_rate = 0.25  # 25% annual carrying cost
            ordering_cost = 50  # Fixed ordering cost
            annual_demand = avg_daily_demand * 365
            
            if annual_demand > 0 and product.standard_price > 0:
                eoq = math.sqrt((2 * annual_demand * ordering_cost) / 
                            (product.standard_price * carrying_cost_rate))
                recommended_qty = max(int(eoq), optimal_reorder - current_stock)
            else:
                recommended_qty = max(0, optimal_reorder - current_stock)
            
            # Urgency calculation
            if current_stock <= 0:
                urgency_score = 1.0
            elif current_stock <= optimal_reorder * 0.5:
                urgency_score = 0.9
            elif current_stock <= optimal_reorder:
                urgency_score = 0.7
            else:
                shortage_ratio = max(0, (optimal_reorder - current_stock) / optimal_reorder)
                urgency_score = shortage_ratio * 0.6
            
            # Adjust urgency based on features
            urgency_score *= features['seasonal_factor']
            urgency_score = min(1.0, urgency_score)
            
            # Generate realistic reasoning
            reasoning_parts = []
            if urgency_score >= 0.9:
                reasoning_parts.append("Critical shortage")
            elif urgency_score >= 0.7:
                reasoning_parts.append("Below reorder point")
            
            if features['seasonal_factor'] > 1.2:
                reasoning_parts.append("seasonal demand increase")
            if features['demand_volatility'] > 0.6:
                reasoning_parts.append("high demand variability")
            if features['supplier_reliability'] < 0.7:
                reasoning_parts.append("supplier reliability concerns")
            if features['stock_coverage_days'] < 15:
                reasoning_parts.append("low stock coverage")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Routine reorder calculation"
            
            return {
                'optimal_reorder': optimal_reorder,
                'urgency_score': urgency_score,
                'recommended_qty': recommended_qty,
                'reasoning': reasoning,
                'estimated_cost': recommended_qty * product.standard_price,
                'lead_time': lead_time_days,
                'safety_stock': int(safety_stock),
                'avg_daily_demand': avg_daily_demand
            }
            
        except Exception as e:
            _logger.warning("Error calculating reorder metrics for product %s: %s", product.id, e)
            return {
                'optimal_reorder': 10,
                'urgency_score': 0.5,
                'recommended_qty': 10,
                'reasoning': "Standard calculation applied",
                'estimated_cost': 10 * (product.standard_price or 0),
                'lead_time': 7,
                'safety_stock': 5,
                'avg_daily_demand': 1.0
            }

    def _calculate_avg_daily_demand(self, product_id):
        """Calculate realistic average daily demand from historical data"""
        try:
            # Look at outgoing moves (sales/deliveries) for the past 90 days
            moves = self.env['stock.move'].search([
                ('product_id', '=', product_id),
                ('state', '=', 'done'),
                ('location_id.usage', '=', 'internal'),
                ('location_dest_id.usage', '!=', 'internal'),
                ('date', '>=', (date.today() - timedelta(days=90)).strftime('%Y-%m-%d'))
            ])
            
            if not moves:
                # If no recent moves, check older data or use minimal default
                older_moves = self.env['stock.move'].search([
                    ('product_id', '=', product_id),
                    ('state', '=', 'done'),
                    ('location_id.usage', '=', 'internal'),
                    ('location_dest_id.usage', '!=', 'internal')
                ], limit=10)
                
                if older_moves:
                    total_qty = sum(move.product_qty for move in older_moves)
                    return max(0.1, total_qty / 90)  # Spread over 90 days
                else:
                    return 0.5  # Minimal default demand
            
            # Calculate daily average from recent moves
            total_demand = sum(move.product_qty for move in moves)
            days_with_data = 90
            
            return max(0.1, total_demand / days_with_data)
            
        except Exception:
            return 0.5  # Safe default

    def _get_realistic_lead_time(self, product_id):
        """Get realistic lead time from supplier or purchase history"""
        try:
            product = self.env['product.product'].browse(product_id)
            
            # Check supplier info first
            if product.seller_ids:
                supplier_delay = product.seller_ids[0].delay or 7
                return max(1, supplier_delay)
            
            # Check recent purchase orders for actual lead times
            po_lines = self.env['purchase.order.line'].search([
                ('product_id', '=', product_id),
                ('order_id.state', 'in', ['purchase', 'done']),
                ('order_id.date_order', '>=', (date.today() - timedelta(days=180)).strftime('%Y-%m-%d'))
            ], limit=5)
            
            if po_lines:
                lead_times = []
                for line in po_lines:
                    if line.order_id.date_order and line.date_planned:
                        lead_time = (line.date_planned.date() - line.order_id.date_order.date()).days
                        if 0 < lead_time <= 90:  # Reasonable lead time
                            lead_times.append(lead_time)
                
                if lead_times:
                    return int(sum(lead_times) / len(lead_times))
            
            return 7  # Default 1 week lead time
            
        except Exception:
            return 7

    def _get_realistic_seasonal_factor(self, product_id):
        """Calculate realistic seasonal factor"""
        try:
            current_month = date.today().month
            
            # Get monthly demand pattern for the past year
            monthly_demand = {}
            for month in range(1, 13):
                start_date = date.today().replace(month=month, day=1) - timedelta(days=365)
                end_date = start_date.replace(day=28) + timedelta(days=4)
                
                moves = self.env['stock.move'].search([
                    ('product_id', '=', product_id),
                    ('state', '=', 'done'),
                    ('location_id.usage', '=', 'internal'),
                    ('location_dest_id.usage', '!=', 'internal'),
                    ('date', '>=', start_date.strftime('%Y-%m-%d')),
                    ('date', '<=', end_date.strftime('%Y-%m-%d'))
                ])
                
                monthly_demand[month] = sum(move.product_qty for move in moves)
            
            # Calculate seasonal factor
            total_demand = sum(monthly_demand.values())
            if total_demand > 0:
                avg_monthly = total_demand / 12
                current_month_demand = monthly_demand.get(current_month, avg_monthly)
                factor = current_month_demand / avg_monthly if avg_monthly > 0 else 1.0
                return max(0.5, min(2.0, factor))  # Cap between 0.5 and 2.0
            
            return 1.0
            
        except Exception:
            return 1.0

    def _calculate_realistic_demand_volatility(self, product_id):
        """Calculate realistic demand volatility using coefficient of variation"""
        try:
            # Get weekly demand for past 12 weeks
            weekly_demands = []
            for week in range(12):
                start_date = date.today() - timedelta(weeks=week+1)
                end_date = date.today() - timedelta(weeks=week)
                
                moves = self.env['stock.move'].search([
                    ('product_id', '=', product_id),
                    ('state', '=', 'done'),
                    ('location_id.usage', '=', 'internal'),
                    ('location_dest_id.usage', '!=', 'internal'),
                    ('date', '>=', start_date.strftime('%Y-%m-%d')),
                    ('date', '<', end_date.strftime('%Y-%m-%d'))
                ])
                
                weekly_demand = sum(move.product_qty for move in moves)
                weekly_demands.append(weekly_demand)
            
            if len(weekly_demands) >= 3:
                mean_demand = statistics.mean(weekly_demands)
                if mean_demand > 0:
                    std_demand = statistics.stdev(weekly_demands)
                    cv = std_demand / mean_demand
                    return min(1.0, cv)  # Cap at 1.0
            
            return 0.4  # Default moderate volatility
            
        except Exception:
            return 0.4

    def _calculate_realistic_lead_time_variance(self, product_id):
        """Calculate lead time variance from actual purchase data"""
        try:
            po_lines = self.env['purchase.order.line'].search([
                ('product_id', '=', product_id),
                ('order_id.state', '=', 'done'),
                ('order_id.date_order', '>=', (date.today() - timedelta(days=365)).strftime('%Y-%m-%d'))
            ])
            
            actual_lead_times = []
            for line in po_lines:
                if (line.order_id.date_order and line.date_planned and 
                    hasattr(line.order_id, 'date_approve') and line.order_id.date_approve):
                    
                    planned_lead_time = (line.date_planned.date() - line.order_id.date_order.date()).days
                    actual_lead_time = (line.order_id.date_approve.date() - line.order_id.date_order.date()).days
                    
                    if 0 < planned_lead_time <= 90 and 0 < actual_lead_time <= 90:
                        variance = abs(actual_lead_time - planned_lead_time) / planned_lead_time
                        actual_lead_times.append(variance)
            
            if actual_lead_times:
                avg_variance = sum(actual_lead_times) / len(actual_lead_times)
                return min(1.0, avg_variance)
            
            return 0.3  # Default 30% variance
            
        except Exception:
            return 0.3

    def _get_urgency_category(self, urgency_score):
        """Convert urgency score to category"""
        if urgency_score >= 0.8:
            return 'Critical'
        elif urgency_score >= 0.6:
            return 'High'
        else:
            return 'Medium'

    def _assess_realistic_supplier_reliability(self, product_id):
        """Assess supplier reliability based on delivery performance"""
        try:
            product = self.env['product.product'].browse(product_id)
            
            if not product.seller_ids:
                return 0.7  # Default reliability
            
            primary_supplier = product.seller_ids[0]
            
            # Get purchase orders from this supplier
            pos = self.env['purchase.order'].search([
                ('partner_id', '=', primary_supplier.partner_id.id),
                ('state', '=', 'done'),
                ('date_order', '>=', (date.today() - timedelta(days=180)).strftime('%Y-%m-%d'))
            ])
            
            if not pos:
                return 0.7  # Default if no history
            
            on_time_count = 0
            total_count = 0
            
            for po in pos:
                for line in po.order_line:
                    if line.product_id.id == product_id:
                        total_count += 1
                        # Check if delivered on time (within planned date)
                        if po.date_approve and po.date_planned:
                            if po.date_approve <= po.date_planned:
                                on_time_count += 1
            
            if total_count > 0:
                reliability = on_time_count / total_count
                return max(0.1, min(1.0, reliability))
            
            return 0.7
            
        except Exception:
            return 0.7

    def _analyze_realistic_price_trends(self, product_id):
        """Analyze price trends using actual purchase data"""
        try:
            po_lines = self.env['purchase.order.line'].search([
                ('product_id', '=', product_id),
                ('order_id.state', 'in', ['purchase', 'done']),
                ('order_id.date_order', '>=', (date.today() - timedelta(days=180)).strftime('%Y-%m-%d'))
            ], order='order_id.date_order')
            
            if len(po_lines) < 2:
                return 0.0  # No trend data
            
            # Get price data points
            price_data = [(line.order_id.date_order, line.price_unit) for line in po_lines]
            price_data.sort(key=lambda x: x[0])
            
            if len(price_data) >= 2:
                first_price = price_data[0][1]
                last_price = price_data[-1][1]
                
                if first_price > 0:
                    price_change = (last_price - first_price) / first_price
                    return max(-0.5, min(0.5, price_change))  # Cap at ±50%
            
            return 0.0
            
        except Exception:
            return 0.0


    def generate_abc_analysis(self):
        """ABC Analysis with expense report styling"""
        try:
            products = self.env['product.product'].search([
                ('type', '=', 'product'),
                ('active', '=', True)
            ])
            
            if not products:
                return '<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4>✅ No Products Found</h4><p>No products available for ABC analysis.</p></div>'
            
            # Calculate annual consumption values
            product_values = []
            total_annual_value = 0.0
            
            for product in products:
                # Get annual consumption
                moves = self.env['stock.move'].search([
                    ('product_id', '=', product.id),
                    ('state', '=', 'done'),
                    ('location_id.usage', '=', 'internal'),
                    ('location_dest_id.usage', '!=', 'internal'),
                    ('date', '>=', (date.today() - timedelta(days=365)).strftime('%Y-%m-%d'))
                ])
                
                annual_qty = sum(move.product_qty for move in moves)
                annual_value = annual_qty * (product.standard_price or 0)
                
                if annual_value > 0:
                    total_annual_value += annual_value
                    product_values.append({
                        'product': product.display_name,
                        'annual_qty': annual_qty,
                        'unit_cost': product.standard_price,
                        'annual_value': annual_value,
                        'current_stock': product.qty_available
                    })
            
            if not product_values:
                return '<div style="background: #fff3cd; padding: 15px; border-radius: 8px; color: #856404; margin: 10px 0; text-align: center;"><h4>⚠️ No Consumption Data</h4><p>No consumption data found for ABC analysis.</p></div>'
            
            # Sort and assign ABC classes
            product_values.sort(key=lambda x: x['annual_value'], reverse=True)
            
            abc_totals = {'A': 0, 'B': 0, 'C': 0}
            cumulative_value = 0
            
            for product in product_values:
                cumulative_value += product['annual_value']
                cumulative_percent = (cumulative_value / total_annual_value) * 100
                
                if cumulative_percent <= 80:
                    product['abc_class'] = 'A'
                elif cumulative_percent <= 95:
                    product['abc_class'] = 'B'
                else:
                    product['abc_class'] = 'C'
                
                abc_totals[product['abc_class']] += product['annual_value']
                product['value_percent'] = (product['annual_value'] / total_annual_value) * 100
            
            # Build HTML
            html_lines = []
            
            # Header
            html_lines.append('<div style="background: linear-gradient(135deg, #6f42c1, #20c997); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;">')
            html_lines.append('<h2 style="margin: 0 0 5px 0; font-size: 18px;"> ABC INVENTORY ANALYSIS</h2>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">💰 Total Annual Consumption: ₹{total_annual_value:,.2f} ({len(product_values)} products)</p>')
            html_lines.append('</div>')
            
            # ABC Class summary
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6f42c1; padding: 8px; margin: 0; font-size: 14px;"> ABC CLASSIFICATION SUMMARY</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">ABC Class</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Annual Value</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
            html_lines.append('</tr></thead><tbody>')
            
            class_colors = {'A': '#28a745', 'B': '#ffc107', 'C': '#dc3545'}
            class_names = {'A': 'A-Class (High Value)', 'B': 'B-Class (Medium Value)', 'C': 'C-Class (Low Value)'}
            
            for i, (abc_class, value) in enumerate([('A', abc_totals['A']), ('B', abc_totals['B']), ('C', abc_totals['C'])]):
                if value > 0:
                    pct = (value / total_annual_value) * 100
                    row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                    color = class_colors[abc_class]
                    
                    html_lines.append(f'<tr style="background-color: {row_bg};">')
                    html_lines.append(f'<td style="padding: 6px; color: #333;">{class_names[abc_class]}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: {color};">₹{value:,.2f}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; color: {color}; font-weight: bold;">{pct:.1f}%</td>')
                    html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #e9ecef; font-weight: bold; border-top: 2px solid #6c757d;">')
            html_lines.append('<td style="padding: 8px; color: #495057;">TOTAL</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #495057;">₹{total_annual_value:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: right; color: #495057;">100.0%</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            # Detailed product table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6c757d; padding: 8px; margin: 0; font-size: 14px;"> DETAILED ABC CLASSIFICATION</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Product</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Class</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Annual Qty</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Unit Cost</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Annual Value</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Current Stock</th>')
            html_lines.append('</tr></thead><tbody>')
            
            # Show top 20 products
            for i, item in enumerate(product_values[:20]):
                # Display full product names without truncation
                name = item['product']
                row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                class_color = class_colors[item['abc_class']]
                
                html_lines.append(f'<tr style="background-color: {row_bg};">')
                html_lines.append(f'<td style="padding: 5px; color: #333; min-width: 250px; word-wrap: break-word;">{name}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center;"><span style="background: {class_color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px;">{item["abc_class"]}</span></td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666; font-weight: bold;">{item["annual_qty"]:.1f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; color: #666;">₹{item["unit_cost"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; font-weight: bold; color: {class_color};">₹{item["annual_value"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #17a2b8; font-weight: bold;">{item["current_stock"]:.0f}</td>')
                html_lines.append('</tr>')
            
            if len(product_values) > 20:
                remaining_count = len(product_values) - 20
                remaining_value = sum(item['annual_value'] for item in product_values[20:])
                html_lines.append('<tr style="background: #fff3cd; font-style: italic;">')
                html_lines.append(f'<td colspan="5" style="padding: 8px; color: #856404;">... and {remaining_count} more products with annual value ₹{remaining_value:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: center; color: #856404;">various</td>')
                html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #e9ecef; font-weight: bold; border-top: 2px solid #6c757d;">')
            html_lines.append(f'<td style="padding: 8px; color: #495057;">TOTAL ({len(product_values)} products)</td>')
            html_lines.append('<td style="padding: 8px; color: #495057;">-</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: center; color: #495057;">{sum(item["annual_qty"] for item in product_values):.1f}</td>')
            html_lines.append('<td style="padding: 8px; color: #495057;">-</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #495057;">₹{total_annual_value:,.2f}</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: center; color: #495057;">{sum(item["current_stock"] for item in product_values):.0f}</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error in ABC analysis: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4>❌ Error</h4><p>{str(e)}</p></div>'


    def _calculate_stock_coverage_days(self, product_id):
        """Calculate how many days current stock will last"""
        try:
            product = self.env['product.product'].browse(product_id)
            current_stock = product.qty_available
            daily_demand = self._calculate_avg_daily_demand(product_id)
            
            if daily_demand > 0:
                coverage_days = current_stock / daily_demand
                return min(999, max(0, coverage_days))  # Cap at 999 days
            
            return 999  # Infinite coverage if no demand
            
        except Exception:
            return 30  # Default 30 days

    def intelligent_reorder_suggestions(self):
        """Reorder suggestions with expense report styling"""
        try:
            products = self.env['product.product'].search([
                ('type', '=', 'product'),
                ('active', '=', True)
            ], limit=50)
            
            if not products:
                return '<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4>✅ No Products Found</h4><p>No products available for reorder analysis.</p></div>'
            
            suggestions = []
            total_investment = 0.0
            urgency_totals = {'Critical': 0, 'High': 0, 'Medium': 0}
            
            for product in products:
                current_stock = product.qty_available
                
                # Calculate reorder metrics (simplified for demo)
                features = self._calculate_enhanced_features(product)
                reorder_metrics = self._calculate_reorder_metrics(product, features)
                
                if reorder_metrics['urgency_score'] >= 0.3:  # Include items needing attention
                    urgency_level = self._get_urgency_category(reorder_metrics['urgency_score'])
                    urgency_totals[urgency_level] += reorder_metrics['estimated_cost']
                    total_investment += reorder_metrics['estimated_cost']
                    
                    suggestions.append({
                        'product': product.display_name,
                        'current_stock': current_stock,
                        'reorder_qty': reorder_metrics['recommended_qty'],
                        'urgency_level': urgency_level,
                        'urgency_score': reorder_metrics['urgency_score'],
                        'estimated_cost': reorder_metrics['estimated_cost'],
                        'lead_time': reorder_metrics['lead_time'],
                        'reasoning': reorder_metrics['reasoning']
                    })
            
            if not suggestions:
                return '<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4>✅ All Stock Levels Good</h4><p>No urgent reorders needed at this time!</p></div>'
            
            # Build HTML using expense format
            html_lines = []
            
            # Header
            html_lines.append('<div style="background: linear-gradient(135deg, #dc3545, #fd7e14); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;">')
            html_lines.append('<h2 style="margin: 0 0 5px 0; font-size: 18px;">&#128680; REORDER RECOMMENDATIONS</h2>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">💰 Total Investment Required: ₹{total_investment:,.2f} ({len(suggestions)} products)</p>')
            html_lines.append('</div>')
            
            # Urgency summary table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #dc3545; padding: 8px; margin: 0; font-size: 14px;">🚨 REORDER PRIORITY BREAKDOWN</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">Priority Level</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Investment</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
            html_lines.append('</tr></thead><tbody>')
            
            priority_order = ['Critical', 'High', 'Medium']
            priority_colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107'}
            
            for i, priority in enumerate(priority_order):
                if urgency_totals[priority] > 0:
                    pct = (urgency_totals[priority] / total_investment) * 100 if total_investment > 0 else 0
                    row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                    color = priority_colors[priority]
                    
                    html_lines.append(f'<tr style="background-color: {row_bg};">')
                    html_lines.append(f'<td style="padding: 6px; color: #333;">{priority} Priority</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: {color};">₹{urgency_totals[priority]:,.2f}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; color: {color}; font-weight: bold;">{pct:.1f}%</td>')
                    html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #ffeaa7; font-weight: bold; border-top: 2px solid #fdcb6e;">')
            html_lines.append('<td style="padding: 8px; color: #e17055;">TOTAL INVESTMENT</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #e17055;">₹{total_investment:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: right; color: #e17055;">100.0%</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            # Detailed reorder table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6c757d; padding: 8px; margin: 0; font-size: 14px;">📋 DETAILED REORDER RECOMMENDATIONS</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px;">Product</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Current</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Order Qty</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Lead Time</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Cost</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Priority</th>')
            html_lines.append('</tr></thead><tbody>')
            
            # Sort by urgency score descending
            suggestions.sort(key=lambda x: x['urgency_score'], reverse=True)
            
            for i, item in enumerate(suggestions[:15]):
                name = item['product'] 
                row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                priority_color = priority_colors.get(item['urgency_level'], '#6c757d')
                
                html_lines.append(f'<tr style="background-color: {row_bg};">')
                html_lines.append(f'<td style="padding: 5px; color: #333;" title="{item["product"]}">{name}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666; font-weight: bold;">{item["current_stock"]:.0f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #28a745; font-weight: bold;">{item["reorder_qty"]:.0f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666;">{item["lead_time"]} days</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; font-weight: bold; color: #dc3545;">₹{item["estimated_cost"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: {priority_color}; font-weight: bold; font-size: 10px;">{item["urgency_level"]}</td>')
                html_lines.append('</tr>')
            
            if len(suggestions) > 15:
                remaining_count = len(suggestions) - 15
                remaining_cost = sum(item['estimated_cost'] for item in suggestions[15:])
                html_lines.append('<tr style="background: #fff3cd; font-style: italic;">')
                html_lines.append(f'<td colspan="5" style="padding: 8px; color: #856404;">... and {remaining_count} more reorder items totaling ₹{remaining_cost:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: center; color: #856404;">various</td>')
                html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #ffeaa7; font-weight: bold; border-top: 2px solid #fdcb6e;">')
            html_lines.append('<td colspan="4" style="padding: 8px; color: #e17055;">TOTAL INVESTMENT REQUIRED</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #e17055;">₹{total_investment:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: center; color: #e17055;">-</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error in reorder suggestions: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4> Error</h4><p>{str(e)}</p></div>'


    def calculate_inventory_turnover(self):
        """Calculate inventory turnover with expense report styling"""
        try:
            products = self.env['product.product'].search([
                ('type', '=', 'product'),
                ('active', '=', True)
            ], limit=50)
            
            if not products:
                return '<div style="background: #d4edda; padding: 15px; border-radius: 8px; color: #155724; margin: 10px 0; text-align: center;"><h4> No Products Found</h4><p>No products available for turnover analysis.</p></div>'
            
            turnover_data = []
            total_cogs = 0.0
            total_avg_inventory = 0.0
            performance_totals = {'Excellent': 0, 'Good': 0, 'Average': 0, 'Poor': 0, 'Very Poor': 0}
            
            for product in products:
                # Calculate Cost of Goods Sold (COGS) for the year
                outgoing_moves = self.env['stock.move'].search([
                    ('product_id', '=', product.id),
                    ('state', '=', 'done'),
                    ('location_id.usage', '=', 'internal'),
                    ('location_dest_id.usage', '!=', 'internal'),
                    ('date', '>=', (date.today() - timedelta(days=365)).strftime('%Y-%m-%d'))
                ])
                
                annual_cogs = sum(move.product_qty * (product.standard_price or 0) for move in outgoing_moves)
                
                if annual_cogs > 0:
                    current_stock_value = product.qty_available * (product.standard_price or 0)
                    avg_inventory_value = current_stock_value  # Simplified calculation
                    
                    if avg_inventory_value > 0:
                        turnover_ratio = annual_cogs / avg_inventory_value
                        days_sales = 365 / turnover_ratio if turnover_ratio > 0 else 365
                        performance = self._classify_turnover_performance(turnover_ratio)
                        
                        performance_totals[performance['class']] += annual_cogs
                        total_cogs += annual_cogs
                        total_avg_inventory += avg_inventory_value
                        
                        turnover_data.append({
                            'product': product.display_name,
                            'annual_cogs': annual_cogs,
                            'avg_inventory_value': avg_inventory_value,
                            'turnover_ratio': turnover_ratio,
                            'days_sales': days_sales,
                            'current_stock': product.qty_available,
                            'performance_class': performance['class'],
                            'performance_color': performance['color']
                        })
            
            if not turnover_data:
                return '<div style="background: #fff3cd; padding: 15px; border-radius: 8px; color: #856404; margin: 10px 0; text-align: center;"><h4> No Movement Data</h4><p>No product movement data found for turnover analysis.</p></div>'
            
            overall_turnover = total_cogs / total_avg_inventory if total_avg_inventory > 0 else 0
            
            # Build HTML using expense format
            html_lines = []
            
            # Header
            html_lines.append('<div style="background: linear-gradient(135deg, #17a2b8, #6610f2); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: center;">')
            html_lines.append('<h2 style="margin: 0 0 5px 0; font-size: 18px;"> INVENTORY TURNOVER ANALYSIS</h2>')
            html_lines.append(f'<p style="margin: 0; font-size: 13px; opacity: 0.9;">Overall Turnover: {overall_turnover:.2f} | Target: 6+ times/year ({len(turnover_data)} products)</p>')
            html_lines.append('</div>')
            
            # Performance summary table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #17a2b8; padding: 8px; margin: 0; font-size: 14px;"&#128202;TURNOVER PERFORMANCE BREAKDOWN</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 8px; text-align: left; font-size: 12px;">Performance Level</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Annual COGS</th>')
            html_lines.append('<th style="padding: 8px; text-align: right; font-size: 12px;">Share</th>')
            html_lines.append('</tr></thead><tbody>')
            
            performance_order = ['Excellent', 'Good', 'Average', 'Poor', 'Very Poor']
            performance_colors = {
                'Excellent': '#28a745', 'Good': '#20c997', 'Average': '#ffc107', 
                'Poor': '#fd7e14', 'Very Poor': '#dc3545'
            }
            
            for i, perf in enumerate(performance_order):
                if performance_totals[perf] > 0:
                    pct = (performance_totals[perf] / total_cogs) * 100 if total_cogs > 0 else 0
                    row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                    color = performance_colors[perf]
                    
                    range_text = {
                        'Excellent': '(12+ turns)', 'Good': '(6-12 turns)', 
                        'Average': '(3-6 turns)', 'Poor': '(1-3 turns)', 'Very Poor': '(<1 turn)'
                    }
                    
                    html_lines.append(f'<tr style="background-color: {row_bg};">')
                    html_lines.append(f'<td style="padding: 6px; color: #333;">{perf} {range_text[perf]}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; font-weight: bold; color: {color};">₹{performance_totals[perf]:,.2f}</td>')
                    html_lines.append(f'<td style="padding: 6px; text-align: right; color: {color}; font-weight: bold;">{pct:.1f}%</td>')
                    html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #cce5ff; font-weight: bold; border-top: 2px solid #17a2b8;">')
            html_lines.append('<td style="padding: 8px; color: #0c5460;">TOTAL ANNUAL COGS</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #0c5460;">₹{total_cogs:,.2f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: right; color: #0c5460;">100.0%</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            # Detailed turnover table
            html_lines.append('<div style="margin: 15px 0; border-radius: 6px; overflow: hidden; border: 1px solid #ddd;">')
            html_lines.append('<h4 style="color: white; background: #6c757d; padding: 8px; margin: 0; font-size: 14px;"> DETAILED TURNOVER ANALYSIS</h4>')
            
            html_lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 12px;">')
            html_lines.append('<thead><tr style="background-color: #f5f5f5;">')
            html_lines.append('<th style="padding: 6px; text-align: left; font-size: 11px; min-width: 300px;">Product</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Turnover Ratio</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Days to Sell</th>')
            html_lines.append('<th style="padding: 6px; text-align: right; font-size: 11px;">Annual COGS</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Current Stock</th>')
            html_lines.append('<th style="padding: 6px; text-align: center; font-size: 11px;">Performance</th>')
            html_lines.append('</tr></thead><tbody>')
            
            # Sort by turnover ratio descending
            turnover_data.sort(key=lambda x: x['turnover_ratio'], reverse=True)
            
            for i, item in enumerate(turnover_data[:15]):
                name = item['product']  # Full product name without truncation
                row_bg = '#f9f9f9' if i % 2 == 0 else 'white'
                perf_color = item['performance_color']
                
                html_lines.append(f'<tr style="background-color: {row_bg};">')
                html_lines.append(f'<td style="padding: 5px; color: #333; min-width: 300px; word-wrap: break-word; white-space: normal;" title="{item["product"]}">{name}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: {perf_color}; font-weight: bold;">{item["turnover_ratio"]:.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666;">{item["days_sales"]:.0f} days</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: right; font-weight: bold; color: #17a2b8;">₹{item["annual_cogs"]:,.2f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: #666; font-weight: bold;">{item["current_stock"]:.0f}</td>')
                html_lines.append(f'<td style="padding: 5px; text-align: center; color: {perf_color}; font-weight: bold; font-size: 10px;">{item["performance_class"]}</td>')
                html_lines.append('</tr>')
            
            if len(turnover_data) > 15:
                remaining_count = len(turnover_data) - 15
                remaining_cogs = sum(item['annual_cogs'] for item in turnover_data[15:])
                html_lines.append('<tr style="background: #fff3cd; font-style: italic;">')
                html_lines.append(f'<td colspan="5" style="padding: 8px; color: #856404;">... and {remaining_count} more products with annual COGS ₹{remaining_cogs:,.2f}</td>')
                html_lines.append('<td style="padding: 8px; text-align: center; color: #856404;">various</td>')
                html_lines.append('</tr>')
            
            # Total row
            html_lines.append('<tr style="background: #cce5ff; font-weight: bold; border-top: 2px solid #17a2b8;">')
            html_lines.append(f'<td style="padding: 8px; color: #0c5460;">TOTAL ({len(turnover_data)} products)</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: center; color: #0c5460;">{overall_turnover:.2f}</td>')
            html_lines.append('<td style="padding: 8px; color: #0c5460;">-</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: right; color: #0c5460;">₹{total_cogs:,.2f}</td>')
            html_lines.append(f'<td style="padding: 8px; text-align: center; color: #0c5460;">{sum(item["current_stock"] for item in turnover_data):.0f}</td>')
            html_lines.append('<td style="padding: 8px; text-align: center; color: #0c5460;">-</td>')
            html_lines.append('</tr>')
            
            html_lines.append('</tbody></table></div>')
            
            # Add interpretation footer
            html_lines.append('<div style="background: #e2e3e5; padding: 12px; border-radius: 6px; margin: 15px 0; color: #383d41; border-left: 4px solid #6c757d;">')
            html_lines.append('<strong> Interpretation:</strong><br>')
            html_lines.append('• <strong>High Turnover (6+):</strong> Efficient inventory management, good demand<br>')
            html_lines.append('• <strong>Low Turnover (<3):</strong> Overstocking, slow-moving inventory, tied-up capital<br>')
            html_lines.append('• <strong>Target:</strong> Most businesses should aim for 6-12 turns per year')
            html_lines.append('</div>')
            
            # Return as Markup to prevent HTML escaping
            from markupsafe import Markup
            return Markup('\n'.join(html_lines))
            
        except Exception as e:
            self.env.cr.rollback()
            _logger.exception("Error in inventory turnover: %s", e)
            return f'<div style="background: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; margin: 10px 0;"><h4> Turnover Analysis Error</h4><p>{str(e)}</p></div>'

    def _classify_turnover_performance(self, turnover_ratio):
        """Classify turnover performance"""
        if turnover_ratio >= 12:
            return {'class': 'Excellent', 'color': '#28a745'}
        elif turnover_ratio >= 6:
            return {'class': 'Good', 'color': '#20c997'}
        elif turnover_ratio >= 3:
            return {'class': 'Average', 'color': '#ffc107'}
        elif turnover_ratio >= 1:
            return {'class': 'Poor', 'color': '#fd7e14'}
        else:
            return {'class': 'Very Poor', 'color': '#dc3545'}
  
    