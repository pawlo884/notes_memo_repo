#!/usr/bin/env python3
"""Test połączenia z DigitalOcean Spaces"""

from dotenv import dotenv_values
import boto3
from botocore.client import Config

# Wczytaj zmienne z .env
env = dotenv_values(".env")

print("=" * 60)
print("TEST POŁĄCZENIA Z DIGITALOCEAN SPACES")
print("=" * 60)

# Sprawdź konfigurację
required_keys = ["DO_SPACES_KEY", "DO_SPACES_SECRET",
                 "DO_SPACES_REGION", "DO_SPACES_BUCKET"]

print("\n1. Sprawdzanie konfiguracji...")
print("-" * 60)

for key in required_keys:
    if key in env and env[key]:
        # Ukryj część wartości dla bezpieczeństwa
        value = env[key]
        if "KEY" in key or "SECRET" in key:
            masked = value[:8] + "..." + \
                value[-4:] if len(value) > 12 else "***"
            print(f"✅ {key}: {masked}")
        else:
            print(f"✅ {key}: {value}")
    else:
        print(f"❌ {key}: BRAK lub PUSTY")

# Sprawdź czy wszystkie klucze są ustawione
missing = [k for k in required_keys if k not in env or not env[k]]
if missing:
    print(f"\n❌ Błąd: Brakujące klucze: {', '.join(missing)}")
    print("\nDodaj te klucze do pliku .env:")
    for key in missing:
        print(f"  {key}=twoja-wartość")
    exit(1)

# Próba połączenia
print("\n2. Próba połączenia z Spaces...")
print("-" * 60)

try:
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=env["DO_SPACES_REGION"],
        endpoint_url=f'https://{env["DO_SPACES_REGION"]}.digitaloceanspaces.com',
        aws_access_key_id=env["DO_SPACES_KEY"],
        aws_secret_access_key=env["DO_SPACES_SECRET"],
        config=Config(signature_version='s3v4')
    )

    print(f"✅ Klient utworzony")
    print(
        f"   Endpoint: https://{env['DO_SPACES_REGION']}.digitaloceanspaces.com")
    print(f"   Region: {env['DO_SPACES_REGION']}")
    print(f"   Bucket: {env['DO_SPACES_BUCKET']}")

    # Sprawdź dostęp do bucketa
    print("\n3. Sprawdzanie dostępu do bucketa...")
    print("-" * 60)

    response = client.list_objects_v2(
        Bucket=env["DO_SPACES_BUCKET"],
        MaxKeys=5
    )

    print(f"✅ Połączenie udane!")

    if 'Contents' in response:
        print(f"   Znaleziono {len(response['Contents'])} plików w buckecie")
        print("\n   Pierwsze pliki:")
        for obj in response['Contents'][:5]:
            size_kb = obj['Size'] / 1024
            print(f"   - {obj['Key']} ({size_kb:.1f} KB)")
    else:
        print("   Bucket jest pusty")

    # Test zapisu
    print("\n4. Test zapisu testowego pliku...")
    print("-" * 60)

    test_content = b"Test file from app_note_v1"
    test_key = "app_note_v1/test_connection.txt"

    client.put_object(
        Bucket=env["DO_SPACES_BUCKET"],
        Key=test_key,
        Body=test_content,
        ACL='public-read',
        ContentType='text/plain'
    )

    url = f'https://{env["DO_SPACES_BUCKET"]}.{env["DO_SPACES_REGION"]}.digitaloceanspaces.com/{test_key}'
    print(f"✅ Plik testowy zapisany!")
    print(f"   URL: {url}")

    # Usuń plik testowy
    client.delete_object(
        Bucket=env["DO_SPACES_BUCKET"],
        Key=test_key
    )
    print(f"✅ Plik testowy usunięty")

    print("\n" + "=" * 60)
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE SUKCESEM!")
    print("=" * 60)

except Exception as e:
    print(f"❌ Błąd: {str(e)}")
    print(f"   Typ błędu: {type(e).__name__}")

    if "InvalidAccessKeyId" in str(e):
        print("\n💡 Sugestie:")
        print("   1. Sprawdź czy DO_SPACES_KEY jest poprawny")
        print("   2. Sprawdź czy DO_SPACES_SECRET jest poprawny")
        print("   3. Upewnij się, że nie ma spacji na początku/końcu kluczy")
        print("   4. Wygeneruj nowe klucze w DigitalOcean → API → Spaces Keys")

    elif "NoSuchBucket" in str(e):
        print("\n💡 Sugestie:")
        print(f"   1. Sprawdź czy bucket '{env['DO_SPACES_BUCKET']}' istnieje")
        print("   2. Sprawdź region - bucket może być w innym regionie")

    print("\n" + "=" * 60)
    print("❌ TEST NIEUDANY")
    print("=" * 60)
    exit(1)
