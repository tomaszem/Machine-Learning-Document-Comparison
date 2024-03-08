# Použij oficiální obraz Pythonu 3.12.0 jako základní obraz
FROM python:3.12.0

# Nastav pracovní adresář v kontejneru
WORKDIR /

# Kopíruj soubory požadavků a nainstaluj závislosti
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Kopíruj celou aplikaci do kontejneru
COPY . .

# Nastav proměnnou prostředí
#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0

# Exponuj port
EXPOSE 5000

# Spusť aplikaci
# CMD ["flask", "run"]

ENTRYPOINT ["python", "./app.py"]