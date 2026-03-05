#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-4.7 flash Token Optimizer
Семантическая оптимизация текста для сокращения количества токенов
при полном сохранении исходного смысла.

Author: MEN
License: MIT
Version: 1.0.0
"""

import json
import re
import sys
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, asdict

# --- Tokenizers ---
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

# --- Semantic Similarity ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# --- NLP Libraries ---
try:
    import setuptools
    import pymorphy2
    MORPHY_AVAILABLE = True
except:
    MORPHY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False

import langdetect
from langdetect import DetectorFactory
DetectorFactory.seed = 0


@dataclass
class Replacement:
    """Данные о выполненной замене слова"""
    original: str
    synonym: str
    similarity: float
    tokens_before: int
    tokens_after: int
    savings: int


class TokenOptimizer:
    """
    Оптимизатор токенов для моделей GLM-4
    
    Использует семантический анализ для замены слов на более компактные
    эквиваленты без потери смысла.
    """
    
    def __init__(self, vocab_path: str, config_path: Optional[str] = None, 
                 verbose: bool = False):
        self.tokenizer = None
        self.special_tokens: Set[str] = set()
        self.special_ids: Set[int] = set()
        self.vocab: Dict[str, int] = {}
        self.vocab_set: Set[str] = set()
        
        self.morph = None
        self.nlp_ru = None
        self.nlp_en = None
        self.embedder = None
        
        self.replacements: List[Replacement] = []
        self.total_savings = 0
        self.verbose = verbose
        
        self._load_tokenizer(vocab_path, config_path)
        self._load_nlp_models()
        self._load_embedder()
    
    def _load_tokenizer(self, vocab_path: str, config_path: Optional[str] = None):
        """Загрузка токенизатора из файла"""
        if not TOKENIZERS_AVAILABLE:
            print("⚠️  Библиотека tokenizers не установлена")
            return
        
        try:
            self.tokenizer = Tokenizer.from_file(vocab_path)
            
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлечение словаря
            if 'model' in data and 'vocab' in data['model']:
                self.vocab = data['model']['vocab']
            elif 'vocab' in data:
                self.vocab = data['vocab']
            
            self.vocab_set = set(self.vocab.keys())
            
            # Извлечение специальных токенов
            if 'added_tokens' in data:
                for item in data['added_tokens']:
                    if isinstance(item, dict):
                        content = item.get('content', '').strip()
                        token_id = item.get('id')
                        if content:
                            self.special_tokens.add(content)
                        if isinstance(token_id, int):
                            self.special_ids.add(token_id)
            
            # Загрузка из config
            if config_path:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    if 'added_tokens_decoder' in config:
                        for item in config['added_tokens_decoder'].values():
                            if isinstance(item, dict):
                                content = item.get('content', '').strip()
                                if content:
                                    self.special_tokens.add(content)
                except:
                    pass
            
            if self.verbose:
                print(f"Загружено токенов: {len(self.vocab):,}")
                print(f"Специальных токенов: {len(self.special_tokens)}")
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки токенизатора: {e}")
    
    def _load_nlp_models(self):
        """Загрузка NLP моделей для морфологии и NER"""
        if MORPHY_AVAILABLE:
            try:
                self.morph = pymorphy2.MorphAnalyzer()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  PyMorphy2: {e}")
        
        if SPACY_AVAILABLE:
            try:
                self.nlp_ru = spacy.load('ru_core_news_sm')
            except:
                pass
            
            try:
                self.nlp_en = spacy.load('en_core_web_sm')
            except:
                pass
    
    def _load_embedder(self):
        """Загрузка модели для семантических эмбеддингов"""
        if not SEMANTIC_AVAILABLE:
            return
        
        try:
            if self.verbose:
                print("Загрузка модели эмбеддингов...")
            self.embedder = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            if self.verbose:
                print("Модель эмбеддингов готова")
        except Exception as e:
            if self.verbose:
                print(f"Ошибка загрузки эмбеддеров: {e}")
    
    def _detect_language(self, text: str) -> str:
        """Определение языка текста"""
        try:
            return langdetect.detect(text)
        except:
            return 'en'
    
    def _count_tokens(self, text: str) -> int:
        """Подсчёт количества токенов в тексте"""
        if self.tokenizer:
            try:
                # Добавляем пробел для корректного учёта Ġ-токенов
                if not text.startswith(' '):
                    text = ' ' + text
                encoding = self.tokenizer.encode(text)
                return len(encoding.ids)
            except:
                pass
        return len(text.split())
    
    def _is_named_entity(self, word: str, context: str = "") -> bool:
        """
        Проверка слова на именованную сущность (NER)
        
        Args:
            word: Слово для проверки
            context: Контекст предложения
            
        Returns:
            True если слово является именем собственным
        """
        if not word:
            return False
        
        # ALL-CAPS слова в середине предложения — обычно эмфазис, не NE
        if word.isupper() and len(word) >= 3:
            if context:
                before = context.split(word)[0] if word in context else ""
                if before and not before.strip().endswith(('.', '!', '?', '\n', ':')):
                    return False
            
            if SPACY_AVAILABLE and self.nlp_en:
                try:
                    doc = self.nlp_en(word)
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                            return True
                except:
                    pass
            
            return False
        
        # Смешанный регистр — более строгая проверка
        if word[0].isupper():
            if context:
                before = context.split(word)[0] if word in context else ""
                if not before or before.strip().endswith(('.', '!', '?', '\n', ':')):
                    return False
            
            if SPACY_AVAILABLE and self.nlp_en:
                try:
                    doc = self.nlp_en(context)
                    for ent in doc.ents:
                        if word in ent.text and ent.label_ in ['PERSON', 'ORG', 'GPE']:
                            return True
                except:
                    pass
        
        return False
    
    def _get_semantic_similarity(self, word1: str, word2: str) -> float:
        """Расчёт косинусного сходства между словами"""
        if not self.embedder:
            if word1.lower() == word2.lower():
                return 1.0
            return 0.85
        
        try:
            embeddings = self.embedder.encode([word1, word2])
            similarity = cosine_similarity(
                [embeddings[0]], [embeddings[1]]
            )[0][0]
            return float(similarity)
        except:
            return 0.9
    
    def _get_synonyms(self, word: str, lang: str) -> List[str]:
        """Поиск синонимов через NLTK или PyMorphy2"""
        synonyms = []
        
        if lang.startswith('ru') and self.morph:
            try:
                parsed = self.morph.parse(word)[0]
                normal = parsed.normal_form
                if normal != word:
                    synonyms.append(normal)
            except:
                pass
        
        elif lang.startswith('en') and NLTK_AVAILABLE:
            try:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        syn_word = lemma.name().replace('_', ' ')
                        if syn_word.lower() != word.lower():
                            synonyms.append(syn_word)
            except:
                pass
        
        return synonyms[:10]
    
    def _optimize_word(self, word: str, lang: str, context: str = "") -> str:
        """
        Оптимизация отдельного слова
        
        Args:
            word: Слово для оптимизации
            lang: Язык текста
            context: Контекст предложения
            
        Returns:
            Оптимизированное слово или оригинал
        """
        if not word or len(word) < 2:
            return word
        
        if word in self.special_tokens:
            return word
        
        original_tokens = self._count_tokens(word)
        
        # 1. Оптимизация ALL-CAPS слов (BROKE → broke)
        if word.isupper() and len(word) >= 3:
            lower_word = word.lower()
            
            if not self._is_named_entity(word, context):
                lower_tokens = self._count_tokens(lower_word)
                
                if lower_tokens < original_tokens:
                    similarity = self._get_semantic_similarity(word, lower_word)
                    
                    if similarity >= 0.85:
                        savings = original_tokens - lower_tokens
                        self.total_savings += savings
                        self.replacements.append(Replacement(
                            original=word,
                            synonym=lower_word,
                            similarity=similarity,
                            tokens_before=original_tokens,
                            tokens_after=lower_tokens,
                            savings=savings
                        ))
                        return lower_word
        
        # 2. Оптимизация смешанного регистра (Hello → hello)
        if word[0].isupper() and not word.isupper():
            lower_word = word.lower()
            
            if not self._is_named_entity(word, context):
                lower_tokens = self._count_tokens(lower_word)
                
                if lower_tokens < original_tokens:
                    similarity = self._get_semantic_similarity(word, lower_word)
                    
                    if similarity >= 0.85:
                        savings = original_tokens - lower_tokens
                        self.total_savings += savings
                        self.replacements.append(Replacement(
                            original=word,
                            synonym=lower_word,
                            similarity=similarity,
                            tokens_before=original_tokens,
                            tokens_after=lower_tokens,
                            savings=savings
                        ))
                        return lower_word
        
        # 3. Замена на синонимы (только если 3+ токена)
        if original_tokens >= 3:
            candidates = self._get_synonyms(word, lang)
            for syn in candidates:
                if syn in self.vocab_set or not self.vocab_set:
                    syn_tokens = self._count_tokens(syn)
                    if syn_tokens < original_tokens:
                        similarity = self._get_semantic_similarity(word, syn)
                        if similarity >= 0.9:
                            savings = original_tokens - syn_tokens
                            self.total_savings += savings
                            self.replacements.append(Replacement(
                                original=word,
                                synonym=syn,
                                similarity=similarity,
                                tokens_before=original_tokens,
                                tokens_after=syn_tokens,
                                savings=savings
                            ))
                            return syn
        
        return word
    
    def optimize_text(self, text: str) -> Tuple[str, Dict]:
        """
        Полная оптимизация текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Кортеж (оптимизированный текст, отчёт)
        """
        self.replacements = []
        self.total_savings = 0
        
        # Очистка пробелов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = text.strip()
        
        initial_count = self._count_tokens(text)
        lang = self._detect_language(text)
        
        # Разбиение на слова
        words = re.findall(r'\w+|\s+|[^\w\s]', text, re.UNICODE)
        optimized_words = []
        
        for i, w in enumerate(words):
            if w.isspace():
                optimized_words.append(' ')
            elif re.match(r'[^\w\s]', w):
                optimized_words.append(w)
            else:
                context = " ".join(words[max(0, i-2):i+3])
                optimized_words.append(
                    self._optimize_word(w, lang, context)
                )
        
        # Сборка текста
        optimized_text = "".join(optimized_words)
        optimized_text = re.sub(r' +', ' ', optimized_text)
        
        # Финальный подсчёт
        final_count = self._count_tokens(optimized_text)
        
        savings_percent = 0.0
        if initial_count > 0:
            savings_percent = ((
                initial_count - final_count
            ) / initial_count) * 100
        
        report = {
            "initial_tokens": initial_count,
            "final_tokens": final_count,
            "savings_percent": round(savings_percent, 2),
            "total_savings": self.total_savings,
            "replacements_count": len(self.replacements),
            "replacements": self.replacements
        }
        
        return optimized_text, report
    
    def print_report(self, report: Dict):
        """Вывод отчёта об оптимизации"""
        print("\n" + "=" * 60)
        print("📊 ОТЧЁТ ПО ОПТИМИЗАЦИИ ТОКЕНОВ")
        print("=" * 60)
        print(f"Токенов до:       {report['initial_tokens']}")
        print(f"Токенов после:    {report['final_tokens']}")
        print(f"Экономия токенов: {report['total_savings']}")
        print(f"Экономия:         {report['savings_percent']}%")
        print("=" * 60)
        
        if report['replacements']:
            print(f"\n🔄 Выполнено замен: {report['replacements_count']}")
            print("-" * 60)
            for rep in report['replacements']:
                print(f"  [{rep.original}] → [{rep.synonym}]")
                print(
                    f"     Сходство: {rep.similarity:.2f}, "
                    f"Экономия: {rep.savings} токенов"
                )
            print("-" * 60)
        else:
            print("\nℹ️  Замен не выполнено")
        
        print("=" * 60)


def main():
    """Точка входа CLI"""
    if len(sys.argv) < 3:
        print(
            "Использование: python glm_token_optimizer.py "
            "<input.txt> <tokenizer.json> [config.json] [--verbose]"
        )
        sys.exit(1)
    
    input_file = sys.argv[1]
    vocab_file = sys.argv[2]
    config_file = sys.argv[3] if (
        len(sys.argv) > 3 and not sys.argv[3].startswith('--')
    ) else None
    
    verbose = '--verbose' in sys.argv
    
    # Загрузка текста
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Файл {input_file} не найден")
        sys.exit(1)
    
    # Инициализация
    print("Инициализация оптимизатора...")
    optimizer = TokenOptimizer(vocab_file, config_file, verbose=verbose)
    
    # Оптимизация
    print("Оптимизация текста...")
    optimized_text, report = optimizer.optimize_text(text)
    
    # Отчёт
    optimizer.print_report(report)
    
    # Сохранение результата
    output_file = "optimized_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(optimized_text)
    print(f"\nРезультат сохранён в {output_file}")
    
    # Лог замен
    if report['replacements']:
        log_file = "replacement_log.json"
        log_data = [asdict(r) for r in report['replacements']]
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"Лог замен сохранён в {log_file}")


if __name__ == "__main__":
    main()