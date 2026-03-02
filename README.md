# App-test-api

Petite application web pour tester plusieurs clés API OpenAI sur une même image, avec le prompt fixe :

> Peux-tu me donner les informations suivantes liées à l'objet principal mis en avant dans la photo que je fournis :  
> Type de matériau  
> Dimensions

## Prérequis

- Node.js 18+

## Installation

```bash
npm install
```

## Lancer

```bash
npm start
```

Puis ouvrir : [http://localhost:3000](http://localhost:3000)

## Utilisation

1. Colle plusieurs clés API (une par ligne).
2. Choisis l'image à analyser.
3. Clique **Lancer le test**.
4. Compare les résultats clé par clé.

## Notes

- Les clés sont utilisées uniquement pour l'appel en cours et ne sont pas stockées.
- Le backend masque les clés dans l'affichage des résultats.# App-test-API_Keys
