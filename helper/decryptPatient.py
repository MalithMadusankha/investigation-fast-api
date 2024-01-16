from cryptography.fernet import Fernet  # type: ignore
import base64


def decryptPatientSensitiveData(patient, en_key):
    print("==== en_key ====", en_key)
    # Convert the key from a string representation to bytes
    ferent_key = base64.urlsafe_b64decode(en_key.encode('utf-8'))
    print("==== key ====", ferent_key)
    # Initialize the Fernet class with the key
    with open('key.key', 'rb') as key:
        secretkey = key.read().strip()
    print("== secretkey ==", secretkey)
    cipher = Fernet(secretkey)

    for key, value in patient.items():
        patient[key] = cipher.decrypt(value.encode()).decode()
    print("==== patient ====", patient)
    return patient
