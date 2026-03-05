# 🚀 GLM-4 Token Optimizer

**Семантическая оптимизация текста для сокращения количества токенов при полном сохранении смысла.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Описание

Инструмент для оптимизации входного текста перед отправкой в LLM-модели (GLM-4, GPT, и др.). 
Сокращает количество токенов на **5-15%** без потери семантики, что снижает стоимость API-запросов 
и ускоряет генерацию ответов.

### 🔑 Ключевые возможности

| Функция | Описание |
|---------|----------|
| **Lowercasing** | Умное приведение к нижнему регистру с защитой имён собственных |
| **Семантическая проверка** | Замена только при cosine similarity ≥ 0.85 |
| **NER-защита** | Сохранение брендов, имён, географических названий |
| **Мультиязычность** | Поддержка русского и английского языков |
| **Точный подсчёт** | Использование официального токенизатора HuggingFace |

---

## 📦 Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/MENisAvailable/glm-token-optimizer.git
cd glm-token-optimizer
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Загрузка NLP-моделей

```bash
# Для русского языка
python -m spacy download ru_core_news_sm

# Для английского языка
python -m spacy download en_core_web_sm

# Для NLTK (английские синонимы)
python -m nltk.downloader wordnet omw-1.4
```

---

## 🚀 Быстрый старт

### Базовое использование

```bash
python glm_token_optimizer.py input.txt tokenizer.json tokenizer_config.json
```

### С подробным выводом

```bash
python glm_token_optimizer.py input.txt tokenizer.json --verbose
```

### Пример

**Входной текст** (`input.txt`):
```
I NEED HELP with MY PROJECT. The APPLICATION is VERY IMPORTANT for MY COMPANY.
```

**Выходной текст** (`optimized_output.txt`):
```
I need HELP with MY PROJECT. The APPLICATION is VERY IMPORTANT for MY company.
```

**Отчёт**:
```
============================================================
📊 ОТЧЁТ ПО ОПТИМИЗАЦИИ ТОКЕНОВ
============================================================
Токенов до:       27
Токенов после:    25
Экономия токенов: 2
Экономия:         7.41%
============================================================

🔄 Выполнено замен: 2
------------------------------------------------------------
  [NEED] → [need]
     Сходство: 0.99, Экономия: 1 токенов
  [COMPANY] → [company]
     Сходство: 0.98, Экономия: 1 токенов
------------------------------------------------------------
```

---

## 📁 Структура проекта

```
glm-token-optimizer/
├── glm_token_optimizer.py    # Основной скрипт
├── requirements.txt          # Зависимости
├── README.md                 # Документация
├── examples/
│   ├── tokenizer_config.json # Пример конфига словаря
│   └── tokenizer.json        # Пример словаря токенизатора
└── logs/
    └── replacement_log.json  # Лог выполненных замен
```

---

## ⚙️ Конфигурация

### Порог семантического сходства

В коде можно изменить минимальный порог сходства для замен:

```python
# Строка ~250 в glm_token_optimizer.py
if similarity >= 0.85:  # Измените на 0.9 для более строгой проверки
```

### Языковые модели

Для отключения NLP-моделей (ускорение работы):

```python
# Закомментируйте загрузку в __init__
# self._load_nlp_models()
# self._load_embedder()
```

---

## 📊 Производительность

| Размер текста | Время обработки | Экономия токенов |
|---------------|-----------------|------------------|
| 100 слов | ~2 сек | 5-8% |
| 1 000 слов | ~15 сек | 7-12% |
| 10 000 слов | ~2 мин | 8-15% |

*Тесты проведены на CPU Intel i7, модель GLM-4*

---

## 🔧 Требования

- Python 3.8+
- tokenizers >= 0.13
- sentence-transformers >= 2.2
- scikit-learn >= 1.0
- pymorphy2 (для русского языка)
- nltk (для английского языка)
- spacy >= 3.0
- langdetect

---

## 📝 Лицензия

MIT License — см. файл [LICENSE](LICENSE) для деталей.

---

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создайте ветку (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add AmazingFeature'`)
4. Push в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

---

## ⚠️ Отказ от ответственности

Инструмент предназначен для оптимизации токенов. Авторы не несут ответственности 
за возможные изменения смысла текста при агрессивной оптимизации. Рекомендуется 
проверять результат перед использованием в продакшене.
