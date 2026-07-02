# Teoria e implementazione della pipeline di `main2.py`

## 1. Obiettivo e struttura generale

`main2.py` orchestra una pipeline di computer vision composta da quattro fasi:

1. **calibrazione intrinseca della camera** (`calibration_phase.py`);
2. **ricostruzione della geometria della pista** (`slope_extraction.py`);
3. **costruzione delle omografie dei piani della pista** (`viz_slope.py`);
4. **stabilizzazione e proiezione 3D della traiettoria del saltatore**
   (`jumper_trajectory.py`).

La pipeline usa il modello di camera pinhole. Un punto 3D in coordinate camera
\(\mathbf{X}=(X,Y,Z)^T\) viene proiettato nel pixel omogeneo
\(\tilde{\mathbf{x}}=(u,v,1)^T\) secondo

\[
\lambda \tilde{\mathbf{x}} = K\mathbf{X},
\qquad
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}.
\]

Questa è la forma della proiezione centrale e della matrice di calibrazione
presentata nelle slide sul sistema ottico e sulla calibrazione intrinseca
[B, p. 26; E, pp. 15, 18-25].

La camera è assunta nell'origine del sistema di riferimento, con orientamento
identità e asse ottico positivo \(+Z\). Lo skew è imposto a zero, mentre
\(f_x\) e \(f_y\) possono essere diversi.

Gli step sono descritti da `PipelineStep`, che contiene nome, funzione da
eseguire e file di output obbligatori. `START_AT` e `STOP_AFTER` selezionano un
sottointervallo contiguo della pipeline. Dopo ogni esecuzione,
`require_outputs()` verifica che gli output esistano e non siano vuoti.
`SKIP_EXISTING=True` evita di ripetere uno step i cui output risultano già
presenti.

La dipendenza tra gli step è:

| Step | Input principale | Output usato dallo step successivo |
|---|---|---|
| `calibration` | frame della pista e gruppi di rette | matrice intrinseca \(K\) |
| `slope` | \(K\), normali annotate e bordi pista | bordi 3D, normale e corrispondenze |
| `viz` | ricostruzione della pista | omografie immagine-piano |
| `trajectory` | video, tracking 2D e omografia del piano centrale | punti stabilizzati e traiettoria sul piano |

Con la configurazione corrente, `START_AT = "calibration"` e
`STOP_AFTER = "trajectory"`, quindi vengono eseguite tutte le fasi.
`WITH_VIEWERS = False` evita l'apertura dei viewer di `viz_slope` e
`jumper_trajectory`, ma non elimina l'annotazione interattiva richiesta da
`slope_extraction.main()`.

Per considerare completato uno step, `main2.py` richiede specificamente:

- `calibration`: `outputGeometry/calibration_K.json`;
- `slope`: `outputGeometry/slope_extraction_annotations.json` e
  `outputGeometry/slope_extraction_reconstruction.npz`;
- `viz`: `outputGeometry/slope_plane_homographies.json` e
  `outputGeometry/slope_plane_points.csv`;
- `trajectory`: `outputGeometry/jumper_trajectory_reference_points.csv` e
  `outputGeometry/jumper_trajectory_3d.npz`.

### 1.1 Convenzione dei riferimenti alle slide

I riferimenti tra parentesi quadre indicano il PDF e la pagina PDF, non
necessariamente il numero grafico mostrato nella slide:

- `[B]`: *IACV 2025 - B - Camera Optical System.pdf*;
- `[C]`: *IACV 2025 - C - 2D Projective Geometry and 2D Reconstruction.pdf*;
- `[D]`: *IACV 2025 - D - 3D Projective Geometry.pdf*;
- `[E]`: *IACV 2025 - E - Single-view Geometry and Camera Calibration.pdf*;
- `[F]`: *IACV 2025 - F - Multi-view Geometry.pdf*;
- `[G]`: *IACV 2025 - G - 3D Reconstruction.pdf*;
- `[H]`: *IACV 2025 - H - Model fitting to noisy images.pdf*.

Le slide motivano i risultati teorici. Parametri numerici, soglie, euristiche
e vincoli aggiuntivi sono invece scelte specifiche dell'implementazione.

## 2. Coordinate omogenee e geometria proiettiva

La pipeline rappresenta punti e rette immagine in coordinate omogenee:

\[
\tilde{\mathbf{x}}=(u,v,1)^T,
\qquad
\mathbf{l}=(a,b,c)^T,
\qquad
\mathbf{l}^T\tilde{\mathbf{x}}=0.
\]

La retta passante per due punti è il loro prodotto vettoriale:

\[
\mathbf{l}=\tilde{\mathbf{x}}_1\times\tilde{\mathbf{x}}_2.
\]

In coordinate non omogenee, per
\(\mathbf{x}_1=(x_1,y_1)\) e \(\mathbf{x}_2=(x_2,y_2)\), il codice costruisce

\[
\mathbf{l} =
\begin{bmatrix}
y_1-y_2 \\
x_2-x_1 \\
x_1y_2-x_2y_1
\end{bmatrix}.
\]

Due rette si intersecano nel punto omogeneo
\(\tilde{\mathbf{v}}=\mathbf{l}_1\times\mathbf{l}_2\). Se molte rette sono
proiezioni di rette 3D parallele, la loro intersezione ideale è un **punto di
fuga**.

Queste operazioni derivano dalla geometria proiettiva 2D: un punto omogeneo è
definito a meno di un fattore non nullo, l'incidenza è
\(\mathbf{l}^T\mathbf{x}=0\), l'intersezione di due rette è il loro prodotto
vettoriale e, per dualità, la retta attraverso due punti è ancora un prodotto
vettoriale [C, pp. 29-30, 42-43, 50, 57-59]. Le slide mostrano inoltre che
rette parallele si incontrano in un punto all'infinito, la cui immagine è il
punto di fuga [C, p. 58; E, pp. 9-10].

## 3. Step `calibration`: stima della matrice intrinseca

### 3.1 Estrazione delle rette

`calibration_phase.main()` legge `FRAME_PATH` e chiama
`preprocess_for_lines()`:

1. conversione BGR-grayscale;
2. filtro gaussiano \(5\times5\), per ridurre rumore ad alta frequenza;
3. CLAHE, per aumentare localmente il contrasto;
4. derivate Sobel \(G_x\) e \(G_y\);
5. modulo del gradiente

   \[
   G=\sqrt{G_x^2+G_y^2};
   \]

6. combinazione pesata tra contrasto e gradiente;
7. Canny con soglie ricavate dalla mediana dell'immagine;
8. closing morfologico \(3\times3\), che connette piccole interruzioni.

La ricerca è limitata alla parte inferiore dell'immagine, a partire da
`LINE_SEARCH_TOP_RATIO = 0.6`. Per impostazione corrente viene usato LSD,
Line Segment Detector. I segmenti sono filtrati per lunghezza, larghezza
stimata e supporto sugli edge. Il supporto è

\[
r_{\text{edge}} =
\frac{\text{campioni del segmento vicini a un edge}}
     {\text{numero totale di campioni}}.
\]

La pipeline conserva al massimo `MAX_LINES = 120` segmenti. I tre gruppi in
`SELECTED_LINE_GROUPS` devono rappresentare tre famiglie di rette 3D
mutuamente ortogonali.

L'eventuale ramo Hough implementa l'idea di voto nello spazio dei parametri
della retta: ogni dato vota per i modelli compatibili e i massimi locali
individuano i modelli dominanti [H, pp. 38-43, 54]. Nella configurazione
corrente questo ramo è disabilitato e viene usato LSD.

### 3.2 Stima robusta dei punti di fuga mediante SVD

Per ogni famiglia viene costruita una matrice

\[
L =
\begin{bmatrix}
\mathbf{l}_1^T \\
\mathbf{l}_2^T \\
\vdots \\
\mathbf{l}_n^T
\end{bmatrix}.
\]

Il punto di fuga ideale soddisfa \(L\tilde{\mathbf{v}}=0\). Con rumore e
segmenti non perfettamente concorrenti si risolve

\[
\min_{\|\tilde{\mathbf{v}}\|=1}\|L\tilde{\mathbf{v}}\|_2.
\]

Per il teorema di Eckart-Young, la soluzione è il vettore singolare destro
associato al più piccolo valore singolare di \(L\). In NumPy è l'ultima riga
di `vh` restituita da `np.linalg.svd(L)`. Il punto viene poi deomogeneizzato:

\[
\mathbf{v} =
\left(
\frac{\tilde v_x}{\tilde v_z},
\frac{\tilde v_y}{\tilde v_z}
\right).
\]

L'implementazione rifiuta punti di fuga quasi all'infinito, cioè con
\(|\tilde v_z|<10^{-9}\).

La soluzione è un problema omogeneo ai minimi quadrati: il vettore cercato è
quello associato al valore singolare minimo. Le slide presentano lo stesso
principio per il fitting con dati rumorosi e sottolineano che l'intersezione
di rette quasi parallele è numericamente poco accurata [H, pp. 19-23;
C, pp. 225-230].

### 3.3 Vincolo di ortogonalità e immagine della conica assoluta

Siano \(\mathbf{d}_1\) e \(\mathbf{d}_2\) due direzioni 3D ortogonali. I loro
punti di fuga soddisfano

\[
\tilde{\mathbf{v}}_1 \sim K\mathbf{d}_1,
\qquad
\tilde{\mathbf{v}}_2 \sim K\mathbf{d}_2.
\]

Poiché \(\mathbf{d}_1^T\mathbf{d}_2=0\),

\[
\tilde{\mathbf{v}}_1^T K^{-T}K^{-1}\tilde{\mathbf{v}}_2=0.
\]

La matrice

\[
\omega=K^{-T}K^{-1}
\]

è l'**immagine della conica assoluta**. È indipendente dalla posa della camera
e permette di ricavare gli intrinseci da vincoli geometrici noti.

Nella notazione delle slide, \(\omega=(KK^T)^{-1}=K^{-T}K^{-1}\). Il vincolo
generale sull'angolo tra due direzioni osservate tramite i rispettivi punti
di fuga diventa lineare quando l'angolo è \(90^\circ\):
\(\mathbf{v}_1^T\omega\mathbf{v}_2=0\) [E, pp. 38-41, 47-48].

Con skew nullo e punto principale noto \((c_x,c_y)\), il vincolo diventa

\[
\frac{(u_1-c_x)(u_2-c_x)}{f_x^2}
+
\frac{(v_1-c_y)(v_2-c_y)}{f_y^2}
+1=0.
\]

Ponendo

\[
a_x=\frac{1}{f_x^2},
\qquad
a_y=\frac{1}{f_y^2},
\]

ogni coppia ortogonale produce un'equazione lineare:

\[
(u_1-c_x)(u_2-c_x)a_x
+
(v_1-c_y)(v_2-c_y)a_y=-1.
\]

`_estimate_zero_skew_k_known_principal_point()` risolve il sistema
sovradeterminato ai minimi quadrati con `np.linalg.lstsq`, quindi calcola

\[
f_x=\sqrt{\frac{1}{a_x}},
\qquad
f_y=\sqrt{\frac{1}{a_y}}.
\]

Se `PRINCIPAL_POINT` non è fornito e `ESTIMATE_PRINCIPAL_POINT=True`, il
codice aggiunge un vincolo metrico basato su due segmenti reali di uguale
lunghezza.

### 3.4 Stima del punto principale con segmenti di uguale lunghezza

I parametri ottimizzati sono

\[
\theta =
\left(\log f_x,\log f_y,\frac{c_x}{M},\frac{c_y}{M}\right),
\qquad
M=\max(\text{larghezza immagine},\text{altezza immagine}).
\]

L'uso di \(\log f_x\) e \(\log f_y\) garantisce focali positive. La funzione
obiettivo contiene:

- i tre residui di ortogonalità tra punti di fuga;
- un residuo che impone uguale lunghezza ai due segmenti sollevati sul piano.

Dal punto di fuga della normale al piano si ricava

\[
\mathbf{n} =
\frac{K^{-1}\tilde{\mathbf{v}}_n}
     {\|K^{-1}\tilde{\mathbf{v}}_n\|}.
\]

Un pixel \(\tilde{\mathbf{x}}\) definisce il raggio
\(\mathbf{r}=K^{-1}\tilde{\mathbf{x}}\). Il codice sceglie il piano canonico
\(\mathbf{n}^T\mathbf{X}=1\); l'intersezione raggio-piano è

\[
\mathbf{X} =
\frac{\mathbf{r}}{\mathbf{n}^T\mathbf{r}}.
\]

Per i due segmenti si calcolano le lunghezze quadrate sollevate
\(L_1^2,L_2^2\), imponendo il residuo normalizzato

\[
r_L =
\frac{L_1^2-L_2^2}{\max(L_1^2,L_2^2,\varepsilon)}.
\]

Il problema non lineare viene risolto con `scipy.optimize.least_squares`,
usando più inizializzazioni e tolleranze `1e-12`. Sono inoltre imposti limiti
di plausibilità su focali e punto principale. L'output è salvato in
`outputGeometry/calibration_K.json`.

Le slide spiegano che la calibrazione intrinseca determina la corrispondenza
tra pixel e direzione del raggio di vista,
\(\mathbf{d}=K^{-1}\tilde{\mathbf{x}}\), e che gli intrinseci non cambiano
con lo spostamento della camera [E, pp. 20, 23, 25]. La stima da segmenti di
uguale lunghezza è una specializzazione progettuale del principio generale:
aggiungere informazione metrica indipendente per eliminare l'ambiguità
proiettiva.

## 4. Step `slope`: ricostruzione dei bordi della pista

### 4.1 Modello geometrico

L'utente annota:

- almeno due segmenti orientati lungo la normale ai piani dei bordi;
- una polilinea per il bordo A;
- una polilinea per il bordo B.

Il modello assunto non considera i due bordi sullo stesso piano. Li considera
invece su due piani paralleli, separati di `TRACK_WIDTH` lungo una normale
\(\mathbf{n}\):

\[
\Pi_A:\mathbf{n}^T\mathbf{X}=d,
\qquad
\Pi_B:\mathbf{n}^T\mathbf{X}=d+w,
\]

dove \(w=\texttt{TRACK_WIDTH}\). Con il valore corrente \(w=1\), la scala è
relativa; diventa metrica solo se `TRACK_WIDTH` è espresso nella reale unità
di misura desiderata.

Per punti corrispondenti sui due bordi viene assunto

\[
\mathbf{X}_{B,i}-\mathbf{X}_{A,i}=w\mathbf{n}.
\]

La rappresentazione
\(\mathbf{n}^T\mathbf{X}=d\) è coerente con la teoria dei piani in geometria
proiettiva 3D, dove i primi tre coefficienti rappresentano la normale e
l'incidenza punto-piano è un prodotto scalare omogeneo [D, pp. 11-14].
L'uso di due piani paralleli richiama inoltre il fatto che piani paralleli
condividono la stessa linea all'infinito [D, pp. 33-35].

### 4.2 Normale del piano da punto di fuga

I segmenti normali annotati vengono convertiti in rette e il loro punto di
fuga \(\tilde{\mathbf{v}}_n\) viene stimato con la stessa soluzione SVD della
fase di calibrazione. La direzione in coordinate camera è

\[
\mathbf{n} =
\frac{K^{-1}\tilde{\mathbf{v}}_n}
     {\|K^{-1}\tilde{\mathbf{v}}_n\|}.
\]

Questo segue direttamente dal modello pinhole: il punto di fuga di una
direzione 3D \(\mathbf{d}\) è proporzionale a \(K\mathbf{d}\).

È precisamente il *Vanishing Point Theorem* delle slide: il raggio di vista
associato al punto di fuga di una direzione è parallelo alla direzione stessa;
per una camera calibrata la direzione si recupera con
\(K^{-1}\tilde{\mathbf{v}}\) [E, pp. 9-11, 25, 32].

### 4.3 Corrispondenze tra i bordi

La polilinea A viene ricampionata uniformemente rispetto alla lunghezza d'arco
immagine. Per vertici \(\mathbf{p}_i\), la lunghezza cumulativa è

\[
s_0=0,
\qquad
s_i=\sum_{k=1}^{i}\|\mathbf{p}_k-\mathbf{p}_{k-1}\|_2.
\]

Le coordinate vengono interpolate in `NUM_SAMPLES = 80` posizioni uniformi
tra \(0\) e \(s_{\max}\). La curva B viene ricampionata densamente con almeno
`DENSE_CURVE_SAMPLES = 1600` punti.

Per ogni punto \(\mathbf{a}_i\) della curva A, la retta di corrispondenza è

\[
\mathbf{l}_i =
\tilde{\mathbf{a}}_i\times\tilde{\mathbf{v}}_n.
\]

Infatti, la proiezione del segmento 3D parallelo a \(\mathbf{n}\) che collega
i due bordi deve passare per il punto di fuga di \(\mathbf{n}\).

La distanza di un candidato \(\mathbf{b}_j=(x_j,y_j)\) dalla retta
\(\mathbf{l}_i=(a_i,b_i,c_i)\) è

\[
d_{ij} =
\frac{|a_ix_j+b_iy_j+c_i|}
     {\sqrt{a_i^2+b_i^2}}.
\]

Il costo aggiunge una regolarizzazione sulla posizione relativa lungo l'arco:

\[
C_{ij} =
d_{ij}
+
\lambda
\left|
\frac{j}{m-1}-\frac{i}{n-1}
\right|,
\qquad
\lambda=\texttt{ARC\_LENGTH\_REGULARIZATION\_PX}=8.
\]

`monotonic_match_curve()` usa programmazione dinamica per trovare una
sequenza di indici strettamente crescente sulla curva B. Questo impedisce
incroci e inversioni locali delle corrispondenze. Il codice prova sia
l'orientamento originale sia quello invertito della curva B e conserva quello
con errore medio punto-retta minore.

### 4.4 Intersezione dei raggi con due piani paralleli

Per ogni pixel corrispondente si costruiscono raggi unitari

\[
\mathbf{r}_{A,i} =
\frac{K^{-1}\tilde{\mathbf{a}}_i}
     {\|K^{-1}\tilde{\mathbf{a}}_i\|},
\qquad
\mathbf{r}_{B,i} =
\frac{K^{-1}\tilde{\mathbf{b}}_i}
     {\|K^{-1}\tilde{\mathbf{b}}_i\|}.
\]

Le intersezioni con i due piani sono

\[
\mathbf{X}_{A,i} =
\frac{d}{\mathbf{n}^T\mathbf{r}_{A,i}}\mathbf{r}_{A,i},
\qquad
\mathbf{X}_{B,i} =
\frac{d+w}{\mathbf{n}^T\mathbf{r}_{B,i}}\mathbf{r}_{B,i}.
\]

Imponendo \(\mathbf{X}_{B,i}-\mathbf{X}_{A,i}=w\mathbf{n}\) si ottiene

\[
d
\left(
\frac{\mathbf{r}_{B,i}}{\mathbf{n}^T\mathbf{r}_{B,i}}
-
\frac{\mathbf{r}_{A,i}}{\mathbf{n}^T\mathbf{r}_{A,i}}
\right)
=
w
\left(
\mathbf{n}
-
\frac{\mathbf{r}_{B,i}}{\mathbf{n}^T\mathbf{r}_{B,i}}
\right).
\]

Tutte le componenti di tutte le coppie formano un sistema lineare
sovradeterminato nell'unica incognita \(d\), risolto ai minimi quadrati.

Questa ricostruzione sfrutta un raggio calibrato e un vincolo geometrico
aggiuntivo. Le slide evidenziano che la calibrazione fornisce la direzione del
raggio, non la profondità del punto lungo il raggio [E, p. 25; F, pp. 2,
13-15]. Qui la profondità viene resa determinabile imponendo appartenenza ai
due piani e separazione nota \(w\).

La direzione di una normale ricavata da un punto di fuga è ambigua a segno:
\(\mathbf{n}\) e \(-\mathbf{n}\) proiettano nello stesso punto di fuga. Il
codice prova entrambi i segni e sceglie prima la soluzione con il maggior
rapporto di profondità positive, poi quella con residuo medio minore. Il
residuo per una coppia è

\[
\mathbf{e}_i =
(\mathbf{X}_{B,i}-\mathbf{X}_{A,i})-w\mathbf{n}.
\]

Gli output principali sono annotazioni, corrispondenze, bordi 3D e normale
del piano.

## 5. Step `viz`: basi planari e omografie

Con `WITH_VIEWERS=False`, `main2.py` non apre il visualizzatore. Carica invece
ricostruzione e annotazioni, chiama `prepare_view_reconstruction()` e salva il
bundle di omografie.

### 5.1 Base ortonormale del piano

Dal primo e ultimo punto del bordo sinistro si ricava una direzione tangente.
La componente lungo la normale viene rimossa tramite proiezione ortogonale:

\[
\mathbf{t}_{\parallel} =
\mathbf{t}-(\mathbf{t}^T\mathbf{n})\mathbf{n}.
\]

La base del piano è quindi

\[
\mathbf{e}_1 =
\frac{\mathbf{t}_{\parallel}}{\|\mathbf{t}_{\parallel}\|},
\qquad
\mathbf{e}_2 =
\frac{\mathbf{n}\times\mathbf{e}_1}
     {\|\mathbf{n}\times\mathbf{e}_1\|}.
\]

Il codice costruisce tre piani paralleli con origini:

\[
\mathbf{o}_A=\mathbf{X}_{A,0},
\qquad
\mathbf{o}_B=\mathbf{o}_A+w\mathbf{n},
\qquad
\mathbf{o}_C=\frac{\mathbf{o}_A+\mathbf{o}_B}{2}.
\]

### 5.2 Omografia piano-immagine

Un punto con coordinate planari \((u_p,v_p)\) corrisponde al punto camera

\[
\mathbf{X}(u_p,v_p) =
\mathbf{o}+u_p\mathbf{e}_1+v_p\mathbf{e}_2.
\]

In forma omogenea:

\[
\mathbf{X} =
\begin{bmatrix}
\mathbf{e}_1 & \mathbf{e}_2 & \mathbf{o}
\end{bmatrix}
\begin{bmatrix}
u_p\\v_p\\1
\end{bmatrix}.
\]

Applicando la proiezione pinhole si ottiene l'omografia

\[
\lambda\tilde{\mathbf{x}}
=
H_{\text{plane}\rightarrow\text{image}}
\begin{bmatrix}
u_p\\v_p\\1
\end{bmatrix},
\qquad
H_{\text{plane}\rightarrow\text{image}}
=
K
\begin{bmatrix}
\mathbf{e}_1 & \mathbf{e}_2 & \mathbf{o}
\end{bmatrix}.
\]

Se il determinante non è quasi nullo, l'omografia inversa è

\[
H_{\text{image}\rightarrow\text{plane}}=H^{-1}.
\]

Vengono costruite omografie per piano A, piano B e piano centrale. Il codice
verifica la ricostruzione tramite errore di riproiezione, errore rispetto ai
piani, separazione normale e residuo di traslazione tra i bordi.

La derivazione coincide con quella delle slide:

\[
H=K
\begin{bmatrix}
\mathbf{w}_{\pi1} & \mathbf{w}_{\pi2} & \mathbf{o}_{\pi}
\end{bmatrix},
\]

dove le prime due colonne descrivono una base del piano e la terza la sua
origine rispetto alla camera [E, pp. 29-31, 65-67]. L'omografia inversa è una
rettificazione del piano: riconduce i punti immagine a coordinate sul piano,
a meno della base e della scala scelte [E, pp. 72-81].

## 6. Step `trajectory`: stabilizzazione e lifting sul piano centrale

### 6.1 Rilevamento dei cambi di camera

Ogni frame viene ridimensionato e convertito in HSV. Si calcola un istogramma
bidimensionale sui canali hue e saturation, normalizzato tra 0 e 1. Due frame
consecutivi vengono confrontati con la distanza di Bhattacharyya.

Per distribuzioni normalizzate \(p\) e \(q\), il coefficiente di
Bhattacharyya è

\[
BC(p,q)=\sum_i\sqrt{p_iq_i},
\]

e una forma comune della distanza è

\[
D_B(p,q)=\sqrt{1-BC(p,q)}.
\]

OpenCV restituisce una distanza vicina a zero per istogrammi simili. Un cambio
è accettato se lo score supera `DEFAULT_DIFF_THRESHOLD = 0.40` e sono passati
almeno `DEFAULT_MIN_SCENE_LEN = 10` frame dall'ultimo cambio.

I cambi dividono il video in intervalli non sovrapposti. Lo step seleziona
`SELECTED_CAMERA_ID` e verifica che contenga `REFERENCE_FRAME`.

### 6.2 Tracking 2D e mascheramento del saltatore

Il CSV fornisce per ogni frame il punto Kalman \((kf_x,kf_y)\) e la bounding
box del saltatore. Il filtro di Kalman è già stato applicato a monte: questa
pipeline legge il risultato, ma non esegue direttamente predizione e
correzione Kalman.

Durante la stima del moto camera, `create_mask()` esclude:

- la bounding box del saltatore;
- regioni fisse contenenti overlay televisivi.

Questa scelta è essenziale: la trasformazione deve descrivere lo sfondo
statico e il moto della camera, non il moto indipendente dell'atleta.

### 6.3 Moto tra frame: SIFT, ratio test e RANSAC

Nonostante la funzione si chiami `ecc_2frame()`, con la configurazione attuale
`GLOBAL_VAR = "ransac"` il ramo realmente eseguito è **SIFT + RANSAC**. ECC
viene usato solo impostando esplicitamente `GLOBAL_VAR = "ecc"`.

Prima della stima viene applicato un texture gate basato sul modulo medio
Sobel. Se il frame non contiene sufficiente struttura, viene restituita
l'identità con score zero.

SIFT rileva keypoint e descrittori invarianti a scala e rotazione. I
descrittori vengono confrontati con distanza euclidea e k-nearest neighbors
con \(k=2\). Il ratio test di Lowe conserva il match \(m_1\) se

\[
\frac{d(m_1)}{d(m_2)}<0.75.
\]

Con `WARP_MODE = cv2.MOTION_HOMOGRAPHY`, RANSAC stima una matrice proiettiva
\(H_{t\rightarrow t-1}\) tale che

\[
\tilde{\mathbf{x}}_{t-1}
\sim
H_{t\rightarrow t-1}\tilde{\mathbf{x}}_t.
\]

Un'omografia ha otto gradi di libertà e richiede almeno quattro
corrispondenze non degeneri. RANSAC campiona modelli minimi, misura l'errore
di riproiezione e seleziona gli inlier entro `ransac_reproj_th = 3.0` pixel.
La configurazione usa fino a 2000 iterazioni e confidenza 0.99.

Nelle slide, RANSAC è presentato come metodo robusto per dati contenenti
outlier: un campione minimo propone un modello, gli altri dati votano tramite
il consensus set e il modello con consenso massimo viene rifinito sugli
inlier [H, pp. 3-8, 37-38, 58-67]. Per un'omografia il campione minimo è
formato da quattro coppie di punti [H, p. 59]. Il numero di campioni necessario
dipende dalla percentuale di inlier e dalla probabilità di successo
desiderata [H, pp. 75-78].

Lo score restituito è il rapporto

\[
s_t =
\frac{\text{numero di inlier}}
     {\text{numero di match accettati}}.
\]

Se l'omografia fallisce, è consentito un fallback affine parziale. Questo
modello descrive traslazione, rotazione e scala uniforme con meno gradi di
libertà, risultando più stabile ma meno espressivo. Se anche il fallback non
è valido, viene usata l'identità.

Il ramo ECC alternativo massimizza l'Enhanced Correlation Coefficient tra
template e immagine trasformata, mediante ottimizzazione iterativa del warp.
Nella configurazione corrente non è però il metodo attivo.

### 6.4 Composizione verso il frame di riferimento

Le trasformazioni consecutive mappano il frame nuovo nel precedente. Per un
frame successivo al riferimento \(r\):

\[
H_{t\rightarrow r}
=
H_{t-1\rightarrow r}H_{t\rightarrow t-1}.
\]

Per i frame precedenti al riferimento si usa l'inversa:

\[
H_{t\rightarrow r}
=
H_{t+1\rightarrow r}
H_{t+1\rightarrow t}^{-1}.
\]

Ogni omografia viene normalizzata dividendo per \(H_{33}\), quando possibile.
Il punto Kalman viene stabilizzato mediante

\[
\tilde{\mathbf{x}}_{r,t}
\sim
H_{t\rightarrow r}\tilde{\mathbf{x}}_t.
\]

Tutti i punti risultano quindi espressi nell'immagine del frame di
riferimento, compensando il moto della camera.

La confidenza cumulativa non è il prodotto degli score, ma il minimo lungo il
cammino:

\[
S_{t\rightarrow r} =
\min_{k\in\text{cammino}(t,r)}s_k.
\]

È una misura conservativa di tipo weakest-link: una singola trasformazione
inaffidabile rende inaffidabile l'intera composizione. Per il plotting, i
punti con score inferiore a `MIN_RELIABLE_CUMULATIVE_SCORE = 0.30` vengono
normalmente esclusi.

### 6.5 Lifting sul piano centrale

Il punto stabilizzato viene trasformato con l'omografia inversa del piano
centrale:

\[
\begin{bmatrix}
u_p\\v_p\\1
\end{bmatrix}
\sim
H_{\text{image}\rightarrow\text{center plane}}
\tilde{\mathbf{x}}_{r,t}.
\]

Le coordinate planari sono convertite in coordinate camera:

\[
\mathbf{P}_t =
\mathbf{o}_C+u_p\mathbf{e}_1+v_p\mathbf{e}_2.
\]

Questa operazione equivale a intersecare il raggio camera del punto con il
piano centrale della pista.

**Conseguenza teorica importante:** il metodo non ricostruisce la vera
posizione 3D libera del saltatore durante il volo. Da una singola camera e un
singolo pixel, senza un ulteriore vincolo, la profondità è indeterminata:
tutti i punti sullo stesso raggio producono lo stesso pixel. La pipeline
risolve l'ambiguità imponendo che ogni punto appartenga al piano centrale.
L'output `trajectory_3d` è quindi la proiezione della traiettoria osservata su
quel piano, non una triangolazione multi-view della traiettoria aerea.

Le slide di multi-view geometry descrivono esattamente l'ambiguità: dopo la
proiezione, profondità diverse lungo lo stesso raggio non sono distinguibili
in una singola immagine [F, p. 2]. La triangolazione richiede invece
l'intersezione di almeno due raggi provenienti da viste e pose distinte
[F, pp. 6-8, 72; G, pp. 7, 13]. In questa pipeline il secondo raggio non
esiste: il piano centrale sostituisce l'informazione multi-view come vincolo
di profondità.

## 7. Assunzioni, degenerazioni e fonti di errore

### Assunzioni principali

- modello pinhole senza distorsione radiale o tangenziale;
- skew nullo;
- tre famiglie di rette realmente ortogonali per la calibrazione;
- segmenti metrici annotati realmente di uguale lunghezza;
- bordi modellabili come curve su due piani paralleli;
- separazione dei bordi costante e nota tramite `TRACK_WIDTH`;
- sfondo sufficientemente statico durante ogni intervallo camera;
- moto inter-frame descrivibile da omografia o trasformazione affine;
- traiettoria finale rappresentabile sul piano centrale.

L'assenza di correzione della distorsione è particolarmente rilevante:
le slide ricordano che i toolbox di calibrazione completi, come quello
OpenCV basato sul metodo di Zhang, stimano anche i parametri di distorsione
[E, pp. 43-46]. In questa pipeline \(K\) viene stimata da una singola
immagine e la distorsione non viene modellata.

### Configurazioni degeneri

- rette quasi parallele nell'immagine possono produrre punti di fuga
  numericamente instabili o all'infinito;
- raggi quasi paralleli ai piani hanno
  \(|\mathbf{n}^T\mathbf{r}|\approx0\), causando profondità instabili;
- una scena poco texturizzata impedisce una stima affidabile con SIFT;
- punti quasi collineari o match concentrati in una piccola regione rendono
  instabile l'omografia;
- errori di annotazione dei bordi alterano corrispondenze, normale, offset dei
  piani e tutte le omografie successive;
- la composizione di molte omografie accumula drift.

Le slide distinguono tra fitting ai minimi quadrati, appropriato con rumore
ma senza outlier, e stima robusta, necessaria quando gli outlier sono presenti
[H, pp. 3-11, 24-38]. La calibrazione e la ricostruzione dei bordi usano
principalmente minimi quadrati e dipendono quindi dalla selezione accurata
delle annotazioni; la stima del moto inter-frame usa invece RANSAC.

### Interpretazione degli errori salvati

- **errore corrispondenze in pixel:** distanza del punto del bordo B dalla
  retta che collega il bordo A al punto di fuga della normale;
- **residuo 3D di traslazione:** deviazione da
  \(\mathbf{X}_B-\mathbf{X}_A=w\mathbf{n}\);
- **positive depth ratio:** frazione di punti davanti alla camera;
- **errore di riproiezione:** distanza in pixel tra punto originale e punto
  3D riproiettato;
- **motion score:** rapporto di inlier RANSAC della trasformazione
  inter-frame;
- **cumulative motion score:** minimo score lungo la composizione verso il
  frame di riferimento.

## 8. Significato complessivo della pipeline

La pipeline combina tre famiglie di vincoli:

1. **vincoli proiettivi**, tramite punti di fuga e omografie;
2. **vincoli metrici**, tramite ortogonalità, segmenti di uguale lunghezza e
   larghezza della pista;
3. **vincoli temporali**, tramite tracking Kalman già disponibile,
   stabilizzazione inter-frame e selezione degli intervalli camera.

La calibrazione trasforma osservazioni puramente immagine in raggi camera. La
ricostruzione della pista usa una larghezza nota per fissare la scala e
recuperare due piani paralleli. Le omografie forniscono coordinate planari
coerenti. Infine, la stabilizzazione elimina il moto camera e consente di
esprimere tutti i punti del saltatore nel frame di riferimento, dove vengono
sollevati sul piano centrale.

Il risultato è geometricamente coerente con le assunzioni introdotte, ma la
sua accuratezza dipende direttamente dalla validità di tali assunzioni e
dalla qualità delle annotazioni, della calibrazione e delle omografie
inter-frame.

## 9. Indice dei principali riferimenti teorici

| Concetto usato nella pipeline | Slide di riferimento |
|---|---|
| Proiezione centrale e matrice \(K\) | [B, p. 26], [E, pp. 15-25] |
| Coordinate omogenee, rette, incidenza e dualità | [C, pp. 29-30, 42-59] |
| Punti all'infinito e punti di fuga | [C, p. 58], [E, pp. 9-10, 32] |
| Raggio calibrato \(K^{-1}\mathbf{x}\) | [E, p. 25] |
| Vincolo ortogonale \(\mathbf{v}_1^T\omega\mathbf{v}_2=0\) | [E, pp. 38-41, 47-48] |
| Piani, normali e parallelismo in 3D | [D, pp. 11-16, 33-35] |
| Omografia piano-immagine | [E, pp. 29-31, 65-67] |
| Rettificazione di un piano calibrato | [E, pp. 72-81] |
| Minimi quadrati e soluzione SVD | [H, pp. 9-23] |
| Hough transform | [H, pp. 38-57] |
| RANSAC e consensus set | [H, pp. 58-78] |
| Ambiguità di profondità monoculare | [F, pp. 2, 13-15] |
| Triangolazione multi-view | [F, pp. 6-8, 72], [G, pp. 7, 13] |
