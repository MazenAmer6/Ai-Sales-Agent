"""
ORM models (SQLAlchemy). One table per business entity, plus two tables that
support the AI agent itself: Conversation/Message (chat history) and
KnowledgeDocument (the human-editable source of truth for RAG — every row
here is also embedded into the Chroma vector store; the dashboard's
"RAG Data Management" screen is really just CRUD on this table, wired to
also update the vector index).
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    brand = db.Column(db.String(100))
    category = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, default=0)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "price": self.price,
            "stock": self.stock,
            "description": self.description,
        }


class Customer(db.Model):
    __tablename__ = "customers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    orders = db.relationship("Order", backref="customer", lazy=True)
    leads = db.relationship("Lead", backref="customer", lazy=True)


class Order(db.Model):
    __tablename__ = "orders"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customers.id"), nullable=False)
    status = db.Column(db.String(50), default="pending")  # pending, confirmed, shipped, cancelled
    total = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    items = db.relationship("OrderItem", backref="order", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "customer_email": self.customer.email if self.customer else None,
            "status": self.status,
            "total": self.total,
            "created_at": self.created_at.isoformat(),
            "items": [
                {
                    "product_id": i.product_id,
                    "product_name": i.product.name if i.product else None,
                    "quantity": i.quantity,
                    "unit_price": i.unit_price,
                }
                for i in self.items
            ],
        }


class OrderItem(db.Model):
    __tablename__ = "order_items"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    unit_price = db.Column(db.Float, nullable=False)

    product = db.relationship("Product")


class Lead(db.Model):
    """A sales inquiry captured by the agent that did not (yet) become an order —
    e.g. the customer asked to be contacted about a bulk purchase or a product
    that's out of stock."""
    __tablename__ = "leads"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customers.id"), nullable=False)
    message = db.Column(db.Text)
    status = db.Column(db.String(50), default="new")  # new, contacted, closed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    customer_email = db.Column(db.String(150), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship("Message", backref="conversation", lazy=True,
                                cascade="all, delete-orphan", order_by="Message.created_at")


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "user" | "assistant" | "tool"
    content = db.Column(db.Text)
    intent = db.Column(db.String(50))  # what the router classified this turn as
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class KnowledgeDocument(db.Model):
    """Source of truth for the RAG knowledge base. Every create/update/delete
    here is mirrored into the Chroma vector store by rag/rag_manager.py, so the
    admin never has to touch the vector DB directly."""
    __tablename__ = "knowledge_documents"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100))  # e.g. "policy", "faq", "shipping"
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "content": self.content,
            "updated_at": self.updated_at.isoformat(),
        }
