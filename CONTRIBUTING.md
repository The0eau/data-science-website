# Contributing Guide

Merci de l'intérêt que vous portez à ce projet ! Que ce soit pour corriger un bug, proposer une nouvelle fonctionnalité ou améliorer la documentation, ce guide vous aidera à démarrer rapidement et en toute sécurité. 🎉

## 🛠️ Configuration de l'environnement

1. **Clonage du dépôt** :

```bash
git clone https://github.com/VOTRE_NOM/NOM_DU_PROJET.git
cd NOM_DU_PROJET

```

2. **Installation des dépendances** :
   _(Adaptez cette ligne selon votre langage : npm install, pip install, etc.)_

```bash
pip install -r requirements.txt

```

3. **Lancement en local** :

```bash
python src/app.py

```

4. **Accès au projet** :
   Visitez `http://localhost:8000` dans votre navigateur pour voir le résultat.

## 🚀 Processus de contribution

1. **Création d'une branche** :
   Utilisez des noms descriptifs et préfixés (ex: `feat/ajout-authentification` ou `fix/correction-bouton`).
2. **Développement et tests** :

- Effectuez vos modifications.
- Assurez-vous que les tests passent localement avant de soumettre.
- **Important** : Ne jamais inclure de données sensibles ou de secrets (clés API) dans votre code.

3. **Soumission (Pull Request)** :
   Poussez votre branche sur GitHub et ouvrez une Pull Request (PR) en utilisant le modèle fourni.
4. **Revue de code** :
   Attendez les retours des mainteneurs et apportez les corrections nécessaires si besoin.

## 📏 Standards de Code (Clean Code)

Pour maintenir une qualité de code optimale, nous suivons ces principes :

- **Style de code** : Respectez les standards du langage (ex: PEP 8 pour Python, Prettier pour JS).
- **Nommage** : Utilisez des noms de variables explicites et descriptifs (ex: `user_account_id` au lieu de `uid`).
- **Documentation** : Commentez les blocs de logique complexes et mettez à jour la documentation si vous modifiez une fonctionnalité existante.
- **Sécurité** : Chaque commit déclenche un scan de sécurité automatique. Assurez-vous qu'aucun avertissement n'est levé.

## 🔒 Branch Protection Rules

To ensure the stability and security of the `main` branch, we enforce the following protection rules:

- **Require a pull request before merging**: All changes must be made through a Pull Request (PR). Direct pushes to `main` are blocked.
- **Require status checks to pass before merging**: All automated checks in the CI pipeline (linting, tests, security scans) must pass before a PR can be merged.
- **Require signed commits**: All commits must be signed with a GPG key to verify the author's identity. This helps prevent commit spoofing.
