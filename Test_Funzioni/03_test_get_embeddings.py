"""TEST 03 — get_embeddings() (llm/embeddings.py)

COSA FA get_embeddings()
------------------------
Costruisce e restituisce il modello di embedding (nomic-embed-text via
Ollama). Un EMBEDDING trasforma il testo in un VETTORE di numeri che ne
cattura il SIGNIFICATO: testi simili -> vettori vicini. Serve sia in
ingestion (per indicizzare i chunk) sia a query time (per trovare i chunk
simili alla domanda). E' @lru_cache: costruito una volta, poi riusato.

COSA VERIFICHIAMO
-----------------
1. che restituisca un OllamaEmbeddings configurato da settings;
2. @lru_cache: due chiamate -> stesso oggetto;
3. embed_query(): un testo -> un vettore (lista di float) da 768 dimensioni;
4. la SEMANTICA: due frasi simili danno vettori piu' vicini (similarita'
   alta) di due frasi scollegate. E' il cuore del perche' il RAG funziona.
"""

from langchain_ollama import OllamaEmbeddings

from copilot.config import settings
from copilot.llm.embeddings import get_embeddings

print("=" * 70)
print("TEST 03 — get_embeddings()")
print("=" * 70)

# --- 1. Tipo e configurazione ---
emb = get_embeddings()
print("\n[1] Tipo restituito:", type(emb))
print("     model    :", emb.model)
print("     base_url :", emb.base_url)
#print("     HAHAHAHAHAHHAHA :", emb)


# --- 2. @lru_cache ---
emb_a = get_embeddings()
emb_b = get_embeddings()
print("\n[2] Effetto @lru_cache:")
print("     stesso oggetto? ->", emb_a is emb_b)

# --- 3. embed_query: testo -> vettore ---
print("\n[3] Vettorizzazione di una frase (embed_query)...")
vettore = emb.embed_query("The pump is vibrating excessively.")
print("     tipo output      :", type(vettore).__name__)
print("     HAHAHAHAHAH      :", type(vettore))
print("     numero dimensioni:", len(vettore))
print("     primi 5 valori   :", [round(x, 4) for x in vettore[:5]])


# --- 4. Semantica: frasi simili -> vettori vicini ---
def similarita_coseno(a, b):
    """Cosine similarity a mano: prodotto scalare / (norma_a * norma_b)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


frase_base    = "The pump is vibrating excessively."
frase_simile  = "The pump shows strong vibrations during operation."
frase_diversa = "I bought fresh strawberries at the market."

v_base    = emb.embed_query(frase_base)
v_simile  = emb.embed_query(frase_simile)
v_diversa = emb.embed_query(frase_diversa)

sim_vicina  = similarita_coseno(v_base, v_simile)
sim_lontana = similarita_coseno(v_base, v_diversa)

print("\n[4] Similarita' semantica (coseno, va da -1 a 1):")
print(f"     base  vs  SIMILE  : {sim_vicina:.4f}  (attesa ALTA)")
print(f"     base  vs  DIVERSA : {sim_lontana:.4f}  (attesa BASSA)")

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

assert isinstance(emb, OllamaEmbeddings)
assert emb.model == settings.embedding_model
assert emb_a is emb_b, "lru_cache non riusa l'istanza!"

# il vettore e' una lista di numeri di dimensione fissa (768 per nomic-embed-text)
assert isinstance(vettore, list)
assert len(vettore) == 768
assert all(isinstance(x, float) for x in vettore)

# la frase simile deve essere PIU' vicina della frase scollegata
assert sim_vicina > sim_lontana, "la semantica non regge: simile non piu' vicina!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: get_embeddings() produce vettori a 768 dim e la")
print("           similarita' semantica funziona. Embeddings OK.")
