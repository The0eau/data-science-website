from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ---------------------------------------------------------
# 1. DATABASE CONFIGURATION (The Engine)
# ---------------------------------------------------------
# 'echo=True' will show you the raw SQL being generated in the console
engine = create_engine('sqlite:///sqlalchemy_demo.db', echo=False)
Base = declarative_base()

# ---------------------------------------------------------
# 2. DEFINING MODELS (The Mirror: Class -> Table)
# ---------------------------------------------------------
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String)
    
    # Relationship: One User can have many Products
    products = relationship("Product", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(username='{self.username}')>"

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    price = Column(Float)
    user_id = Column(Integer, ForeignKey('users.id')) # Link to User table
    
    owner = relationship("User", back_populates="products")

# Create all tables physically in the .db file
Base.metadata.create_all(engine)

# ---------------------------------------------------------
# 3. THE SESSION (The Workspace)
# ---------------------------------------------------------
Session = sessionmaker(bind=engine)
session = Session()

# ---------------------------------------------------------
# 4. CRUD OPERATIONS
# ---------------------------------------------------------

def run_demo():
    print("--- 1. CREATE ---")
    # We create Python objects, and SQLAlchemy handles the INSERT
    new_user = User(username="johndoe", email="john@example.com")
    
    # Adding related data easily
    new_user.products = [
        Product(name="Laptop", price=1200.00),
        Product(name="Mouse", price=25.50)
    ]
    
    session.add(new_user)
    session.commit() # Save to database
    print(f"Added user: {new_user.username} with {len(new_user.products)} products.")

    print("\n--- 2. READ (Query) ---")
    # Get the first user named 'johndoe'
    user = session.query(User).filter_by(username="johndoe").first()
    for product in user.products:
        print(f"Item: {product.name} | Price: ${product.price}")

    print("\n--- 3. UPDATE ---")
    # Find a product and change its price
    laptop = session.query(Product).filter_by(name="Laptop").first()
    if laptop:
        laptop.price = 1100.00 # Just change the attribute!
        session.commit()
        print(f"New price for {laptop.name}: ${laptop.price}")

    print("\n--- 4. DELETE ---")
    # Delete the user (and their products due to 'cascade')
    session.delete(user)
    session.commit()
    print("User and associated products deleted.")

if __name__ == "__main__":
    run_demo()
    session.close()