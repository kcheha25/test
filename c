// Remplissage de la matrice I et tabtime
for (u = 0; u < T; u++) {
    line = sr->Readline();             // Lecture d'une ligne
    tab = line->Split('\t');           // Découpe sur les tabulations

    double jd = (double)u / period;
    j = (int)floor(jd);                // Ligne dans la matrice (axe j)

    double id = (double)u - (double)j * period;
    i = (int)floor(id);                // Colonne dans la matrice (axe i)

    if (j < H && i < W && i >= 0 && j >= 0) {
        tabtime(W - i - 1, j) = getdoublefromstr(tab[0]);

        for (k = 1; k <= Z; k++) {
            if (k < tab->length) {
                I(k - 1, W - i - 1, j) = getdoublefromstr(tab[k]);
            }
        }
    }
}

// Interpolation des zéros dans I par moyenne verticale de voisins
for (j = 0; j < H; j++) {
    for (i = 0; i < W; i++) {
        for (k = 0; k < Z; k++) {
            if (I(k, i, j) == 0) {
                double sum = 0;
                int count = 0;

                // Pixel au-dessus
                if (j > 0 && I(k, i, j - 1) != 0) {
                    sum += I(k, i, j - 1);
                    count++;
                }

                // Pixel en dessous
                if (j < H - 1 && I(k, i, j + 1) != 0) {
                    sum += I(k, i, j + 1);
                    count++;
                }

                // On applique la moyenne si on a trouvé au moins un voisin
                if (count > 0) {
                    I(k, i, j) = sum / count;
                }
            }
        }
    }
}



for (j = 0; j < H; j++) {
    for (i = 0; i < W; i++) {
        for (k = 0; k < Z; k++) {
            if (I(k, i, j) == 0) {
                double sum = 0;
                int count = 0;

                // Parcours des voisins 3D dans une fenêtre de 3x3x3 autour du point (k,i,j)
                for (int dj = -1; dj <= 1; dj++) {
                    for (int di = -1; di <= 1; di++) {
                        for (int dk = -1; dk <= 1; dk++) {
                            // Ne pas prendre le point lui-même
                            if (dk == 0 && di == 0 && dj == 0) continue;

                            int nk = k + dk;
                            int ni = i + di;
                            int nj = j + dj;

                            // Vérification que le voisin est dans les bornes
                            if (nk >= 0 && nk < Z &&
                                ni >= 0 && ni < W &&
                                nj >= 0 && nj < H) {

                                double val = I(nk, ni, nj);
                                if (val != 0) {
                                    sum += val;
                                    count++;
                                }
                            }
                        }
                    }
                }

                // On applique la moyenne si on a trouvé des voisins valides
                if (count > 0) {
                    I(k, i, j) = sum / count;
                }
            }
        }
    }
}



F2D I;
int T = 0;

// Compter le nombre total de lignes
while (sr->Readline()) {
    T++;
}

W = period;
H = T / W;
int TT = H * W;

Resize(I, H, W);

// Initialiser à 0
for (u = 0; u < TT; u++) {
    I(u) = 0;
}

// Réinitialiser le lecteur
sr->Reset();

// Lire la première ligne
line = sr->Readline();
tab = line->Split('\t');

int nbC = tab->length - 1;
int column = /* colonne à utiliser */;
if (column > nbC) column = nbC;
if (column <= 0) column = 1;

int pos;

for (u = 0; u < T; u++) {
    line = sr->Readline();
    tab = line->Split('\t');

    pos = u + offset;
    j = pos / W;
    i = pos - j * W;

    if (j >= 0 && j < H && i >= 0 && i < W) {
        I(W - i - 1, j) = getdoublefromstr(tab[column]);
    }
}

// Interpolation des zéros avec moyenne des voisins 8-connectés
for (j = 0; j < H; j++) {
    for (i = 0; i < W; i++) {
        if (I(i, j) == 0) {
            double sum = 0;
            int count = 0;

            // Balayage des voisins autour de (i, j)
            for (int dj = -1; dj <= 1; dj++) {
                for (int di = -1; di <= 1; di++) {
                    if (di == 0 && dj == 0) continue; // Ne pas prendre le centre

                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < W && nj >= 0 && nj < H) {
                        double val = I(ni, nj);
                        if (val != 0) {
                            sum += val;
                            count++;
                        }
                    }
                }
            }

            if (count > 0) {
                I(i, j) = sum / count;
            }
        }
    }
}


    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            int jj = j + offset;
            while (jj < 0) jj += H;
            while (jj >= H) jj -= H;

            // ligne wrapée ?
            if (jj != j + offset) {
                // appliquer un shift vers la droite sur les colonnes de cette ligne
                // récupérer la ligne dans un buffer temporaire
                std::vector<float> row(W);
                for (int ii = 0; ii < W; ii++) {
                    row[ii] = IM(k, j, ii);
                }

                // faire le shift à droite
                std::rotate(row.rbegin(), row.rbegin() + 1, row.rend());

                // écrire dans la ligne cible jj
                for (int ii = 0; ii < W; ii++) {
                    R(k, jj, ii) = row[ii];
                }
            } else {
                // pas wrapée → copie directe
                R(k, jj, i) = IM(k, j, i);
            }
        }
    }

// Applique le wrap vertical pour décaler les lignes
for (i = 0; i < W; i++) {
    for (j = 0; j < H; j++) {
        int jj = j + offset;

        // Gestion du wrap vertical
        while (jj < 0) jj += H;
        while (jj >= H) jj -= H;

        RR(jj, i) = IM(j, i); 
    }
}

// Applique le décalage horizontal sur les colonnes après le wrap
for (j = 0; j < H; j++) {
    int new_offset = offset % W;  // Calcul du décalage horizontal modulo largeur

    // Si l'offset est positif, on déplace vers la droite
    if (new_offset > 0) {
        for (i = 0; i < W; i++) {
            int new_i = (i + new_offset) % W;  // Calcul du nouvel indice de colonne
            RR(j, new_i) = IM(j, i);
        }
    }
    // Si l'offset est négatif, on déplace vers la gauche
    else if (new_offset < 0) {
        for (i = 0; i < W; i++) {
            int new_i = (i + new_offset + W) % W;  // Calcul du nouvel indice de colonne pour le décalage à gauche
            RR(j, new_i) = IM(j, i);
        }
    }
}

for (i = 0; i < W; i++) {
    for (j = 0; j < H; j++) {
        int jj = j + offset;

        while (jj < 0) jj += H;
        while (jj >= H) jj -= H;

        float val = IM(j, i);

        // Si la ligne a été wrapée (i.e. vient du bas), on décale la colonne d’un cran à droite
        int ii = i;
        if (j + offset < 0 || j + offset >= H) {
            ii = (i + 1) % W; // décale la colonne d’un cran à droite avec wrap
        }

        RR(jj, ii) = val;
    }
}

for (i = 0; i < W; i++) {
    for (j = 0; j < H; j++) {
        int jj = j + offset;

        while (jj < 0) jj += H;
        while (jj >= H) jj -= H;

        // Vérifie si la ligne vient du bas (i.e. a été wrapée)
        bool is_wrapped = (j + offset < 0 || j + offset >= H);

        // On récupère la valeur
        float val = IM(j, i);

        // Si ligne wrapée → on décale les colonnes vers la droite d’un cran
        int ii = is_wrapped ? (i + W - 1) % W : i;

        RR(jj, ii) = val;
    }
}

const double w_direct = 1.0;

for (int iter = 0; iter < iterations; iter++) {
    auto I_copy = I;

    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            for (int k = 0; k < Z; k++) {
                if (I(k, i, j) == 0) {
                    double sum = 0;
                    double weight_sum = 0;

                    // Coordonnées des voisins périodiques haut et bas
                    int j_up    = (j - 1 + H) % H;
                    int j_down  = (j + 1) % H;

                    // Haut
                    if (I(k, i, j_up) != 0) {
                        sum += w_direct * I(k, i, j_up);
                        weight_sum += w_direct;
                    }

                    // Bas
                    if (I(k, i, j_down) != 0) {
                        sum += w_direct * I(k, i, j_down);
                        weight_sum += w_direct;
                    }

                    if (weight_sum > 0) {
                        I_copy(k, i, j) = sum / weight_sum;
                    }
                }
            }
        }
    }

    // Mettre à jour I pour la prochaine itération
    I = I_copy;
}
