from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "Искусственный интеллект становится все более важным в современном мире. Он используется в различных областях, таких как медицина, финансы и транспорт. Машинное обучение и нейронные сети позволяют системам обрабатывать большие объемы данных и делать предсказания с высокой точностью. Инновации в этой сфере продолжают развиваться, открывая новые возможности для бизнеса и общества в целом.",
    "Современные технологии меняют подход к обучению и образованию. Онлайн-курсы и платформы для самообразования становятся все более популярными, позволяя людям учиться в удобное время и в удобном формате. Виртуальная реальность и интерактивные элементы делают процесс обучения более увлекательным и эффективным. Будущее образования выглядит многообещающе благодаря интеграции технологий."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
tfidf_array = tfidf_matrix.toarray()

similarity_text = cosine_similarity(tfidf_array)

print("Косинусное сходство между текстами:")
print(similarity_text)

# Анализируем результат
threshold = 0.5  # Порог сходства
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if similarity_text[i][j] >= threshold:
            print(f"Тексты {i + 1} и {j + 1} похожи (сходство: {similarity_text[i][j]:.2f})")
        else:
            print(f"Тексты {i + 1} и {j + 1} не похожи (сходство: {similarity_text[i][j]:.2f})")