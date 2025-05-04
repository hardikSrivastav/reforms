from app.database import engine
from app.models import Base, SurveyInsightsCache

def create_tables():
    """
    Create all tables in the database if they don't exist.
    This includes the newly added SurveyInsightsCache table.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    create_tables() 