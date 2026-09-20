"""
Admin dashboard. Two jobs, both required by the assessment:
1. Read-only visibility into business data (products, orders, customers, leads,
   conversations) so the admin can see what the agent has been doing.
2. Full CRUD on the RAG knowledge base — every write here calls RAGManager so
   the vector index used by retrieval is always in sync with what's shown on
   screen (the whole point being: you can change what the agent knows without
   touching code).
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash

from models import db, Product, Customer, Order, Lead, Conversation, KnowledgeDocument
from rag.rag_manager import get_rag_manager

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/admin")


@dashboard_bp.route("/")
def home():
    stats = {
        "products": Product.query.count(),
        "orders": Order.query.count(),
        "customers": Customer.query.count(),
        "leads": Lead.query.count(),
        "knowledge_docs": KnowledgeDocument.query.count(),
        "conversations": Conversation.query.count(),
    }
    recent_orders = Order.query.order_by(Order.created_at.desc()).limit(5).all()
    recent_leads = Lead.query.order_by(Lead.created_at.desc()).limit(5).all()
    return render_template("dashboard/home.html", stats=stats,
                            recent_orders=recent_orders, recent_leads=recent_leads)


# ---------------- Products ----------------

@dashboard_bp.route("/products")
def products():
    return render_template("dashboard/products.html", products=Product.query.all())


@dashboard_bp.route("/products/new", methods=["GET", "POST"])
def product_new():
    if request.method == "POST":
        p = Product(
            name=request.form["name"], brand=request.form.get("brand"),
            category=request.form["category"], price=float(request.form["price"]),
            stock=int(request.form.get("stock", 0)), description=request.form.get("description", ""),
        )
        db.session.add(p)
        db.session.commit()
        flash("Product created.", "success")
        return redirect(url_for("dashboard.products"))
    return render_template("dashboard/product_form.html", product=None)


@dashboard_bp.route("/products/<int:product_id>/edit", methods=["GET", "POST"])
def product_edit(product_id):
    p = Product.query.get_or_404(product_id)
    if request.method == "POST":
        p.name = request.form["name"]
        p.brand = request.form.get("brand")
        p.category = request.form["category"]
        p.price = float(request.form["price"])
        p.stock = int(request.form.get("stock", 0))
        p.description = request.form.get("description", "")
        db.session.commit()
        flash("Product updated.", "success")
        return redirect(url_for("dashboard.products"))
    return render_template("dashboard/product_form.html", product=p)


@dashboard_bp.route("/products/<int:product_id>/delete", methods=["POST"])
def product_delete(product_id):
    p = Product.query.get_or_404(product_id)
    db.session.delete(p)
    db.session.commit()
    flash("Product deleted.", "success")
    return redirect(url_for("dashboard.products"))


# ---------------- Orders / Customers / Leads / Conversations (read-only) ----------------

@dashboard_bp.route("/orders")
def orders():
    return render_template("dashboard/orders.html", orders=Order.query.order_by(Order.created_at.desc()).all())


@dashboard_bp.route("/customers")
def customers():
    return render_template("dashboard/customers.html", customers=Customer.query.all())


@dashboard_bp.route("/leads")
def leads():
    return render_template("dashboard/leads.html", leads=Lead.query.order_by(Lead.created_at.desc()).all())


@dashboard_bp.route("/conversations")
def conversations():
    convos = Conversation.query.order_by(Conversation.created_at.desc()).all()
    return render_template("dashboard/conversations.html", conversations=convos)


# ---------------- RAG knowledge base CRUD ----------------

@dashboard_bp.route("/knowledge")
def knowledge():
    return render_template("dashboard/knowledge.html", docs=KnowledgeDocument.query.all())


@dashboard_bp.route("/knowledge/new", methods=["GET", "POST"])
def knowledge_new():
    if request.method == "POST":
        doc = KnowledgeDocument(
            title=request.form["title"], category=request.form.get("category", "general"),
            content=request.form["content"],
        )
        db.session.add(doc)
        db.session.commit()
        get_rag_manager().add_document(doc.id, doc.title, doc.category, doc.content)
        flash("Knowledge added and indexed.", "success")
        return redirect(url_for("dashboard.knowledge"))
    return render_template("dashboard/knowledge_form.html", doc=None)


@dashboard_bp.route("/knowledge/<int:doc_id>/edit", methods=["GET", "POST"])
def knowledge_edit(doc_id):
    doc = KnowledgeDocument.query.get_or_404(doc_id)
    if request.method == "POST":
        doc.title = request.form["title"]
        doc.category = request.form.get("category", "general")
        doc.content = request.form["content"]
        db.session.commit()
        get_rag_manager().update_document(doc.id, doc.title, doc.category, doc.content)
        flash("Knowledge updated and re-indexed.", "success")
        return redirect(url_for("dashboard.knowledge"))
    return render_template("dashboard/knowledge_form.html", doc=doc)


@dashboard_bp.route("/knowledge/<int:doc_id>/delete", methods=["POST"])
def knowledge_delete(doc_id):
    doc = KnowledgeDocument.query.get_or_404(doc_id)
    get_rag_manager().delete_document(doc.id)
    db.session.delete(doc)
    db.session.commit()
    flash("Knowledge deleted from DB and vector index.", "success")
    return redirect(url_for("dashboard.knowledge"))
