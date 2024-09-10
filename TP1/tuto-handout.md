# Utilisation de handout

Voici les étapes que vous allez devoir répéter pour chaque laboratoire.

  1. Créer une équipe: `teamup`
  2. Télécharger votre travail: `fetch`
  3. Soumettre pour une évaluation: `submit`
  4. Obtenir le résultat d'une évaluation: `status`
  5. Voir la note: `result`

D'autres commandes sont disponibles.  Par exemple, il est possible d'afficher la
liste des travaux du cours avec `list` et d'annuler une soumission avec
`cancel`.  La plupart des commandes demandent un jeton d'identification obtenu
à la suite de la commande `teamup`.

## help

Afficher l'aide de l'outil `handout`.

```sh
$ ./handout help
Usage: ./handout COMMAND ARGS...

COMMAND must be one of the sub-commands listed below:

cancel TOKEN                       cancel computation of last submission
fetch TOKEN                        download work for team with TOKEN
help                               print this message
list                               list informations about available works
result TOKEN                       print results of team with TOKEN
status TOKEN                       print status of current computation of team with TOKEN
submit TOKEN ARCHIVE               submit ARCHIVE of team with TOKEN for evaluation
teamup WORK ID1 [ID2]              create a new team for WORK
```

## list

Afficher la liste des travaux du cours.

```sh
$ ./handout list
name: labo-1
scheduled: Tue Sep 06 00:00:00Z 2022
deadline: Wed Sep 28 00:00:00Z 2022

name: labo-2
scheduled: Tue Oct 18 00:00:00Z 2022
deadline: Wed Nov 09 00:00:00Z 2022

name: labo-3
scheduled: Tue Nov 15 00:00:00Z 2022
deadline: Wed Dec 07 00:00:00Z 2022
```

## teamup

C'est la première étape pour commencer un travail.  Il faut passer comme
arguments le nom du travail tel qu'affiché par la commande `list` et les
matricules des étudiants.  Il est possible de former une équipe seul en passant
un seul matricule.  Dans ce cas, vous formerez quand même une équipe de deux
mais avec vous même.

Vous pouvez répéter cette commande autant de fois, vous allez obtenir le même
jeton qui forme votre équipe pour un travail.  Une fois une équipe faite, il
n'est pas possible de changer d'équipe.

```sh
$ ./handout teamup labo-1 1234567
Here's your team token: wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1
```

## fetch

Une fois que vous avez votre jeton pour votre travail, il faut télécharger votre
version du travail.

```sh
$ ./handout fetch wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1
Receiving file labo-1.tar.gz
```

Pour extraire l'archive faites:

```sh
$ mkdir labo-1
$ tar xf labo-1.tar.gz -C labo-1
```

## submit

Vous allez par la suite travailler sur vos postes de travail et itérer sur
ceux-ci.  Vous allez ensuite soumettre pour évaluation votre travail.  Il n'y a
pas de limite d'essai pour l'évaluation.  C'est la note la plus haute qui
compte.  Cependant, il est important d'itérer localement et de soumettre le
moins souvent possible pour éviter d'abuser le noeud de calcul (vous le partagez
tous).

Pour soumettre votre travail, il vous faut générer une archive avec `make
remise` à partir du dossier `build` (ou bien où vous avez fait la commande
`cmake`).

```sh
$ mkdir labo-1/build
$ cd labo-1/build
$ cmake ..
...
$ make remise
...
cd -
$ ./handout submit wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1 ./labo-1/remise.tar.gz
Sending work for build ...
=== BEGIN GUIX ===
...
===  END GUIX  ===
Build successful.
Schedule successful.
Job #91 is in state: PENDING => 0
```

## status

Vous pouvez obtenir l'état de votre soumission à tout moment.  Il est impératif
de faire cette commande pour avoir votre note.  Le système ne détecte en effet
pas que votre travail et terminé.  C'est donc à vous de faire cette commande de
temps en temps après une soumission.

```sh
[oldiob@l4712-05 ~] $ ./handout status wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1
Here's your build hash: z9bszgd6y5kq8v0ncj2rd0rgg3ns2mis
work: labo-1
success: true
serial: 129.25
pthread: 11.90
tbb: 10.25
grade: 20
```

Le `build hash` n'est pas important, mais il est unique et fait référence aux
résultats.  Ainsi, je vous recommande de noter ce hash si vous êtes satisfaits
des résultats.

## cancel

Vous ne pouvez pas soumettre plusieurs fois en même temps.  Si vous voulez
soumettre pour évaluation une nouvelle version de votre travail, il vous faut
soit attendre que l'ancienne version est terminée et faire un `status` pour
obtenir le résulat, ou bien faire la commande `cancel`.

```sh
$ ./handout cancel wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1
Computation canceled.
```

## result

Afficher vos résultats.

```sh
$ ./handout result wkGggc1MnRlzulOfvY8eEaACTfyaRSvb4LWP0vF1
attempt: 4
best-result: 20
last-result: 0
```

