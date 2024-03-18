from ravendb import DocumentStore, documents

# Nastavení DocumentStore
store = DocumentStore(urls=["http://localhost:8080"], database="Results")
store.initialize()

# Definice třídy pro dokument
class User:
    def __init__(self, name, age, city):
        self.Name = name
        self.Age = age
        self.City = city

# Vytvoření a uložení dokumentu
with store.open_session() as session:
    user = User("FirstName LastName", 30, "Prague")
    session.store(user)
    session.save_changes()

print("Success!")
