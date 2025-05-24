import requests
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re


def map_function(text_chunk):
    words = re.findall(r'\b\w+\b', text_chunk.lower())
    return [(word, 1) for word in words]

def shuffle_function(mapped_values):
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()

def reduce_function(shuffled_values):
    reduced = {}
    for key, values in shuffled_values:
        reduced[key] = sum(values)
    return reduced

def map_reduce_parallel(text, num_workers=4):
    lines = text.splitlines()
    chunk_size = len(lines) // num_workers
    chunks = ['\n'.join(lines[i*chunk_size:(i+1)*chunk_size]) for i in range(num_workers)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        mapped_results = list(executor.map(map_function, chunks))

    merged = [pair for sublist in mapped_results for pair in sublist]
    shuffled = shuffle_function(merged)
    reduced = reduce_function(shuffled)

    return reduced

def visualize_top_words(word_freq, top_n=10):
    top_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:top_n]
    words, frequencies = zip(*top_words)

    plt.figure(figsize=(10, 6))
    plt.barh(words, frequencies, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 10 Most Frequent Words')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def get_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Запуск
if __name__ == '__main__':
    url = "https://www.gutenberg.org/files/1342/1342-0.txt" 
    text = get_text_from_url(url)
    result = map_reduce_parallel(text)
    visualize_top_words(result)
