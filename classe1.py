"""
pip install nltk

"""

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Assicurati che nltk abbia i dati necessari
nltk.download('stopwords')
nltk.download('wordnet')

# Funzione di preprocessing del testo
def preprocess_text(text):
    # Convertire il testo in minuscolo
    text = text.lower()
    
    # Rimuovere stopwords
    stop_words = set(stopwords.words('italian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatizzazione
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Generazione del dataset (come nel tuo codice, con più frasi)
frasi_positive = [
    'Adoro questo film!', 'Un capolavoro assoluto!', 'Bellissimo, lo guarderei ogni giorno!',
    'Una storia davvero toccante.', 'Personaggi ben sviluppati e trama coinvolgente.', 'Mi è piaciuto moltissimo!',
    'Lo consiglio vivamente!', 'Uno dei migliori film che abbia mai visto.', 
    'Un’esperienza cinematografica incredibile!', 'Davvero emozionante e ben fatto.', 
    'Ogni minuto del film è stato fantastico!', 'Non riuscivo a staccare gli occhi dallo schermo.', 
    'Un’opera che ti lascia senza parole.', 'Non vedo l’ora di guardarlo di nuovo!', 
    'Davvero una produzione impeccabile.', 'Un cast stellare per un film straordinario.', 
    'La colonna sonora è semplicemente perfetta!', 'Un mix di emozioni che ti travolge.', 
    'La regia è eccezionale, un vero maestro.', 'Un film che tutti dovrebbero vedere almeno una volta.', 
    'Superiore a ogni aspettativa.', 'Non posso fare a meno di consigliarlo a tutti.', 
    'Dialoghi brillanti e scene memorabili.', 'Il messaggio del film è potente e ispirante.', 
    'Una perla rara nel panorama cinematografico.', 'Coinvolgente dall’inizio alla fine.', 
    'Un viaggio emozionante e unico.', 'Un vero capolavoro del cinema moderno.', 
    'Tutti dovrebbero vederlo, è straordinario.', 'Ho riso, pianto e sono rimasto incantato.', 
    'Ogni scena è curata nei minimi dettagli.', 'Un film che resta nel cuore.', 
    'Non dimenticherò mai questa storia.', 'Uno spettacolo visivo incredibile.', 
    'Il miglior film che ho visto quest’anno.', 'Un’avventura indimenticabile.', 
    'Un film che ti fa riflettere profondamente.', 'Non riuscivo a smettere di pensare a quanto fosse bello.', 
    'Davvero un film magistrale.', 'La performance degli attori è stata straordinaria.', 
    'Mi ha lasciato senza fiato.', 'Un capolavoro emozionante e unico.', 
    'Perfetto sotto ogni punto di vista.', 'Un’esperienza cinematografica che ti segna.', 
    'Non ho parole per descrivere quanto mi sia piaciuto!', 'Semplicemente spettacolare!',
    'Un film che tocca il cuore di chi lo guarda.', 'Una delle migliori storie mai raccontate.', 
    'Un film che ti fa sentire vivo.', 'Ogni minuto è stato emozionante e coinvolgente.','che bel film.',
    'film bello, salvo qualche scena. Ci siamo.','film che trasmette tranquillita.Genuino, semplice e che rispetta la realta del passato.'
]

frasi_negative = [
    'Pessimo film, non lo guarderò mai più.', 'Una delusione totale.', 'Non vale la pena di guardarlo.',
    'Sceneggiatura banale e recitazione scadente.', 'Una perdita di tempo.', 'Non lo consiglierei a nessuno.',
    'Molto noioso, ho perso interesse dopo 10 minuti.', 'Troppo prevedibile e senza emozioni.',
    'Un vero disastro, non so come abbiano potuto farlo uscire.', 'Recitazione pessima e trama senza senso.',
    'Che schifo di film! Non poteva essere peggio.', 'Buttati nel cesso due ore della mia vita.', 
    'Non ho mai visto niente di più stupido.', 'Un insulto al cinema, davvero una schifezza.', 
    'Hanno rovinato una buona idea con questa porcata.', 'Effetti speciali ridicoli, sembra un film degli anni 80.',
    'Ma chi diavolo ha scritto questa roba?', 'Il peggior film che abbia mai visto, senza dubbio.', 
    'Una vera cagata, non ci sono altre parole.','che cacata di film.', 'Avrei fatto meglio a guardare la vernice che si asciuga.',
    'Ridicolo in ogni singolo dettaglio.', 'Dialoghi assurdi e recitazione da cani.', 
    'Non mi aspettavo nulla, ma è riuscito comunque a deludermi.', 'Che film di merda, mai più.', 
    'Un completo fallimento sotto ogni punto di vista.', 'Che cavolata immensa, non ci posso credere.', 
    'Era meglio stare a casa e dormire.', 'Sembra fatto da dilettanti allo sbaraglio.', 
    'Non capisco come qualcuno possa apprezzare questa schifezza.', 'Ho sprecato i miei soldi e il mio tempo.', 
    'Un vero strazio dall’inizio alla fine.', 'Semplicemente orribile.', 'Assurdo, un pasticcio totale.', 
    'Una trama così stupida che fa quasi ridere.', 'Effetti speciali patetici, sembra un gioco per bambini.', 
    'Non so chi abbia trovato questo film interessante, ma io l’ho odiato.', 'Che porcheria assurda!', 
    'Un film che non merita nemmeno di essere commentato.', 'Si sono davvero impegnati a fare un film così brutto.', 
    'Non riesco a credere che qualcuno abbia pensato fosse una buona idea.', 'Completamente inutile e insignificante.', 
    'Non potevo credere a quanto fosse terribile.', 'Ho dovuto spegnerlo a metà, era inguardabile.', 
    'Un pasticcio di cattivo gusto, senza senso e senza stile.', 'Una cagata pazzesca!', 'Una merda totale!','ti lascia con l amaro in bocca.',
    'Un film che non vale nemmeno il tempo di scrivere una recensione.', 'Non capisco come possa avere buone recensioni.',
    'Mi chiedo come sia possibile fare un film così brutto.', 'Un film che ti fa pentire di averlo visto.','che brutto film.',
    'film brutto, salvo qualche scena. non ci siamo.','film che trasmette inquietudine.Poco genuino. Non rispecchia la realta del passato.'
]

frasi_neutre = [
    'Il film è ok, niente di speciale.', 'Non mi ha emozionato particolarmente.', 'Una pellicola media.',
    'Non male, ma neanche eccezionale.', 'Un film discreto, nulla più.', 'Semplicemente nella norma.',
    'Non spicca né in positivo né in negativo.', 'Una visione accettabile per passare il tempo.',
    'Si lascia guardare, ma non mi ha colpito.', 'Va bene per una serata senza pretese.', 
    'Un film così così, niente di che.', 'Non è brutto, ma non mi ha lasciato nulla.', 
    'Mah, carino ma dimenticabile.', 'Un film nella media, non saprei cos’altro dire.', 
    'Boh, niente di nuovo sotto il sole.', 'Non è né carne né pesce.', 'Un film normale, senza infamia e senza lode.', 
    'Non so, è passabile ma nulla di speciale.', 'Alla fine, non mi ha né deluso né entusiasmato.', 
    'Un lavoro mediocre, ma si può guardare.', 'Non è un capolavoro, ma nemmeno una porcheria.', 
    'È guardabile, ma non lo riguarderei.', 'Semplicemente un film come tanti altri.', 
    'Beh, almeno non mi sono annoiato del tutto.', 'Non è sto granché, ma va bene per spegnere il cervello.', 
    'Un film che non ti cambia la vita, ma non ti fa nemmeno incazzare.', 'Che dire? È ok, ma non fa gridare al miracolo.', 
    'Non mi ha entusiasmato, ma neanche fatto schifo.', 'Bah, un film normale, né bello né brutto.', 
    'Guardabile, ma niente di memorabile.', 'Un film senza troppe pretese, diciamo che ci sta.', 
    'Non è un capolavoro, ma nemmeno una cagata.', 'Un film che ti fa dire "meh".', 
    'Onestamente, poteva essere meglio, ma non è da buttare.', 'Non è una bomba, ma nemmeno una schifezza.', 
    'Una pellicola normale, non ti fa incazzare ma nemmeno applaudire.', 'Alla fine, non lascia un gran segno.', 
    'Mah, un film come tanti altri, niente di che.', 'Non mi ha entusiasmato, ma neanche annoiato.', 
    'Un lavoro senza grandi difetti, ma anche senza grandi pregi.', 'Che dire? Un film passabile e basta.', 
    'Ok per ammazzare il tempo, ma non aspettarti granché.', 'Alla fine della fiera, un film normale.', 
    'Semplicemente guardabile, nulla di più.', 'Un film che si lascia vedere, ma senza pretese.', 
    'Non fa schifo, ma non lo riguarderei.', 'Un lavoro che non sorprende né delude troppo.',
    'Un film che si dimentica facilmente.', 'Un’esperienza cinematografica che non lascia il segno.'
]



# Creazione del dataset
dataset = []
for _ in range(1000):  # 1000 campioni
    sentimento = random.choice(['positivo', 'negativo', 'neutro'])
    if sentimento == 'positivo':
        frase = random.choice(frasi_positive)
    elif sentimento == 'negativo':
        frase = random.choice(frasi_negative)
    else:
        frase = random.choice(frasi_neutre)
    
    dataset.append([preprocess_text(frase), sentimento])

# Creazione del DataFrame
df = pd.DataFrame(dataset, columns=['Testo', 'Sentimento'])

# Salvataggio del dataset
df.to_csv('dataset_sentimenti.csv', index=False)

# Preprocessing e suddivisione del dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Testo'])  # Trasformazione del testo in vettori numerici
y = df['Sentimento']

# Creazione del modello
modello = MultinomialNB()
modello.fit(X, y)

# Creazione dell'app Flask
app = Flask(__name__)

# Route principale
@app.route('/')
def index():
    return render_template('index.html')  # Una pagina HTML che mostrerò più avanti

# Route per il chatbot
@app.route('/chat', methods=['POST'])
def chat():
    commento = request.json.get('commento')
    commento_trasf = vectorizer.transform([preprocess_text(commento)])
    
    # Predizione del sentiment
    commento_predetto = modello.predict(commento_trasf)
    
    # Risposta in base al sentiment
    if commento_predetto[0] == 'positivo':
        risposta = "Sembra che ti sia piaciuto! ????"
    elif commento_predetto[0] == 'negativo':
        risposta = "Mi dispiace che non ti sia piaciuto. ?? Cosa non ti è piaciuto?"
    else:
        risposta = "Non sono sicuro di cosa pensi. ?? Forse è stato ok?"
    
    return jsonify({'risposta': risposta})

if __name__ == '__main__':
    app.run(debug=True)


    
    





