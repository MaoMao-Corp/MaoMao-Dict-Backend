import torch
import numpy as np 
from transformers import (
    AutoTokenizer,
    AutoModel,
    DebertaV2Tokenizer,
    DebertaV2Model
)

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity 
import spacy
from functools import lru_cache 
import warnings 
from itertools import chain

from maps import UD_TO_SIMPLE 
from html_parse import get_definitions
from Errors import TargetPositionError

warnings.filterwarnings("ignore")

class WordSenseDisambiguator:

    def __init__(self, use_fp16: bool = True):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device == "cuda"

        print("Initializing model on", self.device, "...")
        if self.use_fp16:
            print("--> FP16 enabled")

        print("Loading bi-encoder")
        self.bi_encoder = SentenceTransformer('all-mpnet-base-v2', device=self.device)

        print("Loading cross-encoder")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=self.device, max_length = 512)

        print("Loading DeBERTa-v3")
        self.deberta_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
        self.deberta_model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to(self.device)
        self.deberta_model.eval()

        if self.use_fp16:
            self.deberta_model = self.deberta_model.half()

        print("Loading spacy transformer-based PoS tagger")
        self.nlp = spacy.load("en_core_web_trf")

        self._definition_cache = {}
        self._context_cache = {}

        self.ensemble_weights = {
            'bi_encoder': 0.35,
            'cross_encoder': 0.65,
            'contextualized': 0.3
        }

        self.pos_map = UD_TO_SIMPLE
        
        print("Loaded succesfully")

    #------- POS SECTION ----------
    def get_pos_tag(self, sentence:str, target_position: int) -> tuple[str, str]:

        doc = self.nlp(sentence)
        if target_position >= len(doc): # fallback if I calculated bad the position, TO BE FIXED
            raise TargetPositionError("Target position lays outside the range of tokenized sentence")
        tkn = doc[target_position]

        return (tkn.lemma_, self.pos_map[tkn.pos_])


    def filter_by_pos(self, entries:dict[str, list[dict]], target_pos:str) -> list[dict]:
        return entries.get(target_pos, list(chain.from_iterable(entries.values())))

    #------- BI ENCODER SECTION -------

    @lru_cache(maxsize=10000)
    def _get_bi_encoder_embedding(self, text:str) -> np.ndarray:
        return self.bi_encoder.encode(text, convert_to_numpy=True)
        
    def bi_encoder_retrieval(self, sentence:str, entries: list[dict]
                             ) -> list[tuple[dict, float]]:

        sentence_emb = self._get_bi_encoder_embedding(sentence)
        definitions = [e['definition'] for e in entries]
        def_emb = []
        for d in definitions:
            if d not in self._definition_cache:
                self._definition_cache[d] = self._get_bi_encoder_embedding(d)
            else: print("CACHE HIT~ >///~///<")
            def_emb.append(self._definition_cache[d])

        def_emb = np.array(def_emb)

        similarities = cosine_similarity([sentence_emb], def_emb)[0]

        ranked_entries = [
            (entry, float(score))
            for entry, score in zip(entries, similarities)
        ]
        ranked_entries.sort(key=lambda x: x[1], reverse=True)
        print(ranked_entries[0][0]['definition'])
        
        return ranked_entries
    

    # ------- CROSS ENCODER SECION ----------
    def cross_encoder_rerank(self, sentence:str, candidates: list[tuple[dict, float]]):

        pairs = [
            [sentence, entry['definition']]
            for entry, _ in candidates
        ]
        
        cross_scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)

        reranked = [
            (entry, bi_score, float(cross_score))
            for (entry, bi_score), cross_score in zip(candidates, cross_scores)
        ]
        
        reranked.sort(key=lambda x: x[2], reverse=True)

        print(reranked[0][0]['definition'])

        return reranked

    def ensemble_fusion(
            self, 
            scored_entries: list[tuple[dict, float, float]]
    ) -> list[dict]:

        results = []

        for entry, bi_score, cross_score in scored_entries:
            final_score = (
                self.ensemble_weights['bi_encoder'] * bi_score + 
                self.ensemble_weights['cross_encoder'] * cross_score
            )

            result = entry.copy()
            
            result['final_score'] = float(final_score)
            result['bi_encoder_score'] = float(bi_score)
            result['cross_encoder_score'] = float(cross_score)

            results.append(result)

        results.sort(key = lambda x: x['final_score'], reverse=True)

        return results
    
    def disambiguate(
        self,
        sentence: str,
        target_position: int,
        verbose: bool = True
    ) -> list[dict]:

        if verbose:
            print(f"\n{'-'*50}")
            print(f"Disambiguating: '{sentence.split()[target_position]}' in context")
            print(f"Sentence: {sentence}")
            print(f"\n{'-'*50}")

        lemma, pos_tag = self.get_pos_tag(sentence, target_position)
        pos_tag:str
        lemma: str

        if verbose:
            print("\tPoS Tag:", pos_tag)

        all_entries: dict[str, list[dict]] = get_definitions(lemma)
        filtered_entries: list[dict] = self.filter_by_pos(all_entries, pos_tag)
        if verbose:
            print(f"\tThere are {len(filtered_entries)} filtered entries")

        candidates_1 = self.bi_encoder_retrieval(sentence, filtered_entries)
        if verbose:
            print("\tBi-encoder done")

        candidates_2 = self.cross_encoder_rerank(sentence, candidates_1)
        if verbose:
            print("\tCross-encoder done")

        final_results = self.ensemble_fusion(candidates_2)
        if verbose:
            print(f"\tEnsemble done\n")
            print(f"{'-'*50}")
            print("FINAL RESULTS:")
            print(f"{'-'*50}\n")
            for i, result in enumerate(final_results, 1):
                print(f"{i}. Score: {result['final_score']:.4f}")
                print(f"   {result['definition'][:100]}...")
                print(f"   (Bi: {result['bi_encoder_score']:.3f} | "
                      f"Cross: {result['cross_encoder_score']:.3f} | "
                    )
                print()
        
        return final_results



if __name__ == "__main__":
    wsd = WordSenseDisambiguator(use_fp16=True)
    while True:
        sentence = input("sentence: ")
        index = int(input("index: "))
        
        wsd.disambiguate(sentence, index, verbose=True)
        



        input("Press enter to return")
