"""
Run once after first setup: `python seed_data.py`
Creates sample products and knowledge-base documents, and indexes the
knowledge documents into the vector store so RAG has something to retrieve
from day one.
"""
from app import create_app
from models import db, Product, KnowledgeDocument
from rag.rag_manager import get_rag_manager

PRODUCTS = [
    dict(name="Aurora X1 Laptop", brand="Aurora", category="Laptops", price=899.99, stock=15,
         description="14-inch laptop, Intel i5, 16GB RAM, 512GB SSD. Great for students and office work."),
    dict(name="Aurora X1 Pro Laptop", brand="Aurora", category="Laptops", price=1299.99, stock=8,
         description="15-inch laptop, Intel i7, 32GB RAM, 1TB SSD, dedicated GPU. Built for creators and gaming."),
    dict(name="Nimbus Buds Pro", brand="Nimbus", category="Audio", price=149.99, stock=40,
         description="Wireless earbuds with active noise cancellation and 30-hour battery life."),
    dict(name="Nimbus Speaker Mini", brand="Nimbus", category="Audio", price=59.99, stock=60,
         description="Compact Bluetooth speaker, waterproof, 12-hour battery."),
    dict(name="Pixelview 27 Monitor", brand="Pixelview", category="Monitors", price=329.99, stock=20,
         description="27-inch 4K IPS monitor, 60Hz, USB-C with 65W power delivery."),
    dict(name="Pixelview 27 Gaming Monitor", brand="Pixelview", category="Monitors", price=449.99, stock=0,
         description="27-inch QHD 165Hz gaming monitor with 1ms response time. Currently out of stock."),
    dict(name="SnapShot Mirrorless Camera", brand="SnapShot", category="Cameras", price=749.00, stock=10,
         description="24MP mirrorless camera with interchangeable lens mount, 4K video."),
    dict(name="ChargeUp 100W GaN Charger", brand="ChargeUp", category="Accessories", price=39.99, stock=100,
         description="Compact 100W USB-C GaN charger, fast-charges laptops and phones."),
]

KNOWLEDGE_DOCS = [
    dict(title="Shipping Policy", category="shipping", content=
         "We ship across the country via standard and express courier.\n"
         "Standard shipping takes 3-5 business days and costs $5.99, free on orders over $75.\n"
         "Express shipping takes 1-2 business days and costs $14.99.\n"
         "Orders placed before 2 PM local time ship the same business day."),
    dict(title="Return & Refund Policy", category="policy", content=
         "You can return most items within 30 days of delivery for a full refund, as long as they are "
         "in original condition with packaging.\n"
         "Opened earbuds/audio items can only be returned if defective, for hygiene reasons.\n"
         "Refunds are issued to the original payment method within 5-7 business days after we receive the return."),
    dict(title="Warranty Information", category="policy", content=
         "All electronics come with a 1-year manufacturer warranty covering defects in materials and workmanship.\n"
         "Laptops and monitors can be upgraded to a 2-year extended warranty for an additional fee at checkout.\n"
         "Warranty does not cover accidental damage, liquid damage, or unauthorized repairs."),
    dict(title="Payment Methods", category="faq", content=
         "We accept all major credit/debit cards, PayPal, and Apple Pay / Google Pay.\n"
         "We also support installment payments (Buy Now Pay Later) on orders over $150 through our partner."),
    dict(title="Store Hours & Contact", category="faq", content=
         "TechHub Electronics customer support is available Monday-Saturday, 9 AM to 8 PM.\n"
         "You can reach us via live chat (this assistant), email at support@techhub.example, "
         "or phone at 1-800-555-0199."),
    dict(title="Order Cancellation", category="policy", content=
         "Orders can be cancelled free of charge within 1 hour of placing them, as long as they haven't shipped.\n"
         "After that, please wait for delivery and use our standard return policy instead."),
]


def run():
    app = create_app()
    with app.app_context():
        if Product.query.count() == 0:
            for p in PRODUCTS:
                db.session.add(Product(**p))
            db.session.commit()
            print(f"Seeded {len(PRODUCTS)} products.")
        else:
            print("Products already exist, skipping.")

        if KnowledgeDocument.query.count() == 0:
            rag = get_rag_manager()
            for d in KNOWLEDGE_DOCS:
                doc = KnowledgeDocument(**d)
                db.session.add(doc)
                db.session.flush()  # get doc.id
                rag.add_document(doc.id, doc.title, doc.category, doc.content)
            db.session.commit()
            print(f"Seeded and indexed {len(KNOWLEDGE_DOCS)} knowledge documents.")
        else:
            print("Knowledge documents already exist, skipping.")


if __name__ == "__main__":
    run()
