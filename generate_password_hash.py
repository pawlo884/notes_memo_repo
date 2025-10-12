"""
Skrypt do generowania hasha hasła dla systemu logowania aplikacji
Użycie: python generate_password_hash.py
"""
from hashlib import sha256
import getpass


def generate_hash(password):
    """Generuje hash SHA256 z hasła"""
    return sha256(password.encode()).hexdigest()


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATOR HASHA HASŁA DLA APLIKACJI NOTE APP")
    print("=" * 60)
    print()

    # Pobierz hasło od użytkownika (nie będzie widoczne podczas wpisywania)
    password = getpass.getpass("Wpisz hasło: ")
    password_confirm = getpass.getpass("Potwierdź hasło: ")

    if password != password_confirm:
        print("\n❌ Hasła nie są identyczne!")
        exit(1)

    if len(password) < 6:
        print("\n❌ Hasło powinno mieć minimum 6 znaków!")
        exit(1)

    # Wygeneruj hash
    password_hash = generate_hash(password)

    print("\n" + "=" * 60)
    print("✅ HASH WYGENEROWANY POMYŚLNIE!")
    print("=" * 60)
    print()
    print("Dodaj poniższe linie do pliku .env:")
    print()
    print(f"APP_USERNAME=admin")
    print(f"APP_PASSWORD_HASH={password_hash}")
    print()
    print("UWAGA: Możesz zmienić 'admin' na dowolny login.")
    print("=" * 60)
