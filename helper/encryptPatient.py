from cryptography.fernet import Fernet  # type: ignore
import base64
aa = Fernet.generate_key()

with open('key.key', 'wb') as key_file:
    key_file.write(aa)


def encryptPatientSensitiveData(patient):
    secretkey = ""
    with open('key.key', 'rb') as key:
        secretkey = key.read().strip()
    # Generate a new encryption key
    # ferent_key = Fernet.generate_key()

    # Initialize the Fernet class with the key
    print("secretkey", secretkey)
    cipher = Fernet(secretkey)

    for key, value in patient.items():
        patient[key] = cipher.encrypt(value.encode()).decode()

    print("==== ferent_key ====", secretkey)
    # Convert the key to a string representation before storing it
    key_str = base64.urlsafe_b64encode(secretkey).decode('utf-8')
    print("==== secretkey ====", secretkey)
    return {"patient": patient, "en_key": key_str}
