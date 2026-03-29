# set_admin.py

from app import app, db, User

with app.app_context():
    db.create_all()  # sécurité — crée les tables si manquantes

    user = User.query.filter_by(email="mikarocco@hotmail.com").first()

    if not user:
        print("Utilisateur non trouve — cree d'abord ton compte sur /register")
    else:
        user.is_admin = True
        db.session.commit()
        print(f"Admin OK -> {user.email}")
        
@app.route("/make-admin")
def make_admin():
    user = User.query.first()
    user.is_admin = True
    db.session.commit()
    return "admin ok"