import re
import math
from collections import Counter

# === 1. Парсинг документа ===
def load_document(path="data/document.md"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Разделяем на текст и изображения
    parts = re.split(r"^## (Текст|Изображения)\s*$", text, flags=re.MULTILINE)

    text_content = ""
    images = []

    current_section = None
    for i, part in enumerate(parts):
        part = part.strip()
        if part == "Текст":
            current_section = "text"
        elif part == "Изображения":
            current_section = "images"
        elif current_section == "text" and part:
            text_content = part
        elif current_section == "images" and part:
            # Парсим изображения вида "### Изображение N: ... \n[Описание: ...]"
            img_blocks = re.split(r"^### (.+)$", part, flags=re.MULTILINE)
            for j in range(1, len(img_blocks), 2):
                title = img_blocks[j].strip()
                body = img_blocks[j + 1].strip() if j + 1 < len(img_blocks) else ""
                match = re.search(r"\[Описание:\s*(.+?)\]", body, re.DOTALL)
                description = match.group(1).strip() if match else body
                images.append({
                    "title": title,
                    "description": description,
                    "source": f"image::{title}"
                })

    return text_content, images

# === 2. Чанкинг текста ===
def chunk_text(text, chunk_size=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "id": f"text::{i}",
            "source": "text",
            "text": p
        })
    return chunks

# === 3. TF-IDF поиск ===
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_index(chunks):
    doc_freq = Counter()
    for c in chunks:
        for t in set(tokenize(c["text"])):
            doc_freq[t] += 1
    return doc_freq

def score(query_tokens, chunk, doc_freq, total_docs):
    tokens = tokenize(chunk["text"])
    tf = Counter(tokens)
    s = 0.0
    for qt in query_tokens:
        if qt in tf:
            idf = math.log((total_docs + 1) / (doc_freq.get(qt, 0) + 1)) + 1
            s += tf[qt] * idf
    return s

def search(query, chunks, doc_freq, top_k=3):
    q_tokens = tokenize(query)
    scored = [(score(q_tokens, c, doc_freq, len(chunks)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# === 4. Генерация ответа (демо) ===
def generate_answer(query, top_results):
    if not top_results or top_results[0][0] == 0:
        return "Не найдено релевантных документов."
    answer = f"Вопрос: {query}\n\nНайденные фрагменты:\n"
    for i, (s, c) in enumerate(top_results, 1):
        answer += f"\n[{i}] Источник: {c['source']} (релевантность: {s:.2f})\n"
        answer += f"    {c['text'][:200]}...\n"
    answer += "\n(Демо-режим. В реальности здесь был бы ответ от LLM на основе найденного контекста.)"
    return answer

# === 5. Запуск ===
if __name__ == "__main__":
    text_content, images = load_document()
    print(f"Загружено текста: {len(text_content)} символов")
    print(f"Загружено изображений: {len(images)}")

    # Чанки текста
    text_chunks = chunk_text(text_content)

    # «Чанки» изображений (используем описания как текст для поиска)
    image_chunks = [
        {"id": f"img::{i}", "source": img["source"], "text": img["description"]}
        for i, img in enumerate(images)
    ]

    all_chunks = text_chunks + image_chunks
    print(f"Всего чанков для индексации: {len(all_chunks)} (текст: {len(text_chunks)}, картинки: {len(image_chunks)})")

    doc_freq = build_index(all_chunks)

    queries = [
        "Какие цели у Google по энергии?",
        "Покажи диаграмму выбросов CO2",
        "Где находятся дата-центры Google?",
        "Как изменилось водопотребление?",
        "Какой PUE у дата-центров Google?",
    ]

    with open("results/answers.txt", "w", encoding="utf-8") as f:
        for q in queries:
            top = search(q, all_chunks, doc_freq, top_k=3)
            ans = generate_answer(q, top)
            print("=" * 60)
            print(ans)
            f.write("=" * 60 + "\n" + ans + "\n\n")

    print("\nРезультаты сохранены в results/answers.txt")
