"""
Tools = the functions the LLM is allowed to call. Each one is decorated with
@tool so LangChain turns its signature + docstring into a JSON schema the
model sees. Two tools (`search_products`, `check_order_status`) are read-only
lookups; `rag_search` hits the vector store; `create_order` is the required
"real business action" — it actually writes Customer/Order/OrderItem rows and
decrements stock in SQLite, it does not just return a canned LLM sentence.

These tools assume they run inside an active Flask app context (they are,
because the whole graph is invoked from inside a Flask request handler —
see routes/chat.py), so `db.session` works normally.
"""
import re
from typing import Optional, List
from langchain_core.tools import tool

from models import db, Product, Customer, Order, OrderItem, Lead
from rag.rag_manager import get_rag_manager

# Deliberately server-side, not left to the LLM: the model can be talked into
# accepting "test@test" or similar as "good enough", so every tool that takes
# a customer_email validates it in code before touching the database.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _invalid_email_msg(email: str) -> str:
    return (f"'{email}' doesn't look like a valid email address (needs a name, an @, "
             f"and a domain with a dot, e.g. jane@example.com). Please ask the customer "
             f"for a correct email before trying again.")


@tool
def search_products(query: str, category: Optional[str] = None, max_price: Optional[float] = None) -> str:
    """Search the product catalog by keyword, optionally filtered by category
    and/or maximum price. Use this whenever the customer asks what products
    are available, wants a recommendation, or asks for a specific item's
    price/stock. Returns a short list of matching products with id, name,
    price and stock."""
    q = Product.query
    if query:
        like = f"%{query}%"
        q = q.filter(db.or_(Product.name.ilike(like), Product.description.ilike(like)))
    if category:
        q = q.filter(Product.category.ilike(f"%{category}%"))
    if max_price is not None:
        q = q.filter(Product.price <= max_price)
    results = q.limit(5).all()
    if not results:
        return "No matching products found."
    return "\n".join(
        f"#{p.id} {p.name} ({p.brand}) - ${p.price:.2f} - stock: {p.stock} - {p.description[:120]}"
        for p in results
    )


@tool
def check_product_availability(product_id: int) -> str:
    """Check the live stock count for a specific product by its id. Use this
    before confirming an order to make sure the item is actually in stock."""
    p = Product.query.get(product_id)
    if not p:
        return f"No product with id {product_id}."
    if p.stock <= 0:
        return f"{p.name} is currently OUT OF STOCK."
    return f"{p.name} has {p.stock} units in stock at ${p.price:.2f} each."


@tool
def rag_search(query: str) -> str:
    """Search the store's knowledge base (FAQs, shipping info, return/warranty
    policy, payment methods, store info) for information relevant to the
    customer's question. Always use this for policy/process questions instead
    of guessing."""
    hits = get_rag_manager().search(query, k=3)
    if not hits:
        return "No relevant knowledge base entries found."
    return "\n\n".join(f"[{h['category']}] {h['title']}: {h['content']}" for h in hits)


@tool
def create_order(customer_email: str, customer_name: str, items: List[dict]) -> str:
    """Place a real order. This is a business action with side effects: it
    creates the customer if new, creates an Order row, creates OrderItem rows,
    and decrements product stock. `items` must be a list of objects like
    {"product_id": 1, "quantity": 2}. Only call this after the customer has
    clearly confirmed they want to buy, and after you've checked stock with
    check_product_availability. Returns the order id and total, or an error
    if a product is unavailable or the email is invalid."""
    if not _EMAIL_RE.match(customer_email or ""):
        return _invalid_email_msg(customer_email)

    customer = Customer.query.filter_by(email=customer_email).first()
    if not customer:
        customer = Customer(email=customer_email, name=customer_name)
        db.session.add(customer)
        db.session.flush()  # get customer.id before commit

    order = Order(customer_id=customer.id, status="confirmed", total=0.0)
    db.session.add(order)
    db.session.flush()

    total = 0.0
    for item in items:
        product = Product.query.get(item["product_id"])
        qty = int(item.get("quantity", 1))
        if not product:
            db.session.rollback()
            return f"Order failed: no product with id {item['product_id']}."
        if product.stock < qty:
            db.session.rollback()
            return f"Order failed: only {product.stock} units of {product.name} available."
        product.stock -= qty
        line_total = product.price * qty
        total += line_total
        db.session.add(OrderItem(order_id=order.id, product_id=product.id,
                                  quantity=qty, unit_price=product.price))

    order.total = total
    db.session.commit()
    return f"Order #{order.id} confirmed for {customer_email}. Total: ${total:.2f}."


@tool
def check_order_status(order_id: Optional[int] = None, customer_email: Optional[str] = None) -> str:
    """Look up existing order(s) either by order id or by customer email. Use
    this when a customer asks 'where is my order' or 'what did I order'."""
    if order_id:
        order = Order.query.get(order_id)
        if not order:
            return f"No order with id {order_id}."
        return f"Order #{order.id}: status={order.status}, total=${order.total:.2f}"
    if customer_email:
        if not _EMAIL_RE.match(customer_email):
            return _invalid_email_msg(customer_email)
        customer = Customer.query.filter_by(email=customer_email).first()
        if not customer or not customer.orders:
            return f"No orders found for {customer_email}."
        return "\n".join(f"Order #{o.id}: status={o.status}, total=${o.total:.2f}" for o in customer.orders)
    return "Please provide an order_id or customer_email."


@tool
def create_lead(customer_email: str, customer_name: str, message: str) -> str:
    """Capture a sales lead when the customer is interested but not ready to
    buy right now (e.g. wants a callback, asked about a bulk/enterprise deal,
    or wants a product that's out of stock). Creates the customer if new.
    Returns an error if the email is invalid."""
    if not _EMAIL_RE.match(customer_email or ""):
        return _invalid_email_msg(customer_email)

    customer = Customer.query.filter_by(email=customer_email).first()
    if not customer:
        customer = Customer(email=customer_email, name=customer_name)
        db.session.add(customer)
        db.session.flush()
    lead = Lead(customer_id=customer.id, message=message, status="new")
    db.session.add(lead)
    db.session.commit()
    return f"Lead #{lead.id} captured for {customer_email}. Our sales team will follow up."


SALES_TOOLS = [search_products, check_product_availability, rag_search, create_order, create_lead]
SUPPORT_TOOLS = [rag_search, check_order_status, search_products]
ALL_TOOLS = [search_products, check_product_availability, rag_search, create_order, check_order_status, create_lead]
