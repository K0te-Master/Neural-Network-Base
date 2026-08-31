# Neural-Network-Base
Простой класс нейросети с полезными методами:
- **set_input_layer** - Установка входного слоя значениями
- **loop** - Цикл активации нейронов
- **network_error** - Возвращает ошибку нейросети
- **back_propagation** - Обратное распространение ошибки

## Пример кода
```c++
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "KoteMaster/network.h"

using std::cout;
using std::cin;
using std::string;
using std::vector;

// Функция для обработки файла нейросетью
void train(Network &network, string filename, int lines_amount, const float &doBackProp) {
    // Файл со значениями и правильными ответами
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "File error!!\n";
        return;
    }
    
    string line_text; // Строка файла
    float error = 0;  // Ошибка нейросети
    int correct = 0;  // Кол-во правильных ответов
    
    for (int line = 0; std::getline(file, line_text); line++) {
        // Объект 1 строки
        json obj = json::parse(line_text);

        vector<float> input = obj.value("image", vector<float>(784)); // Входные значения [0-255]
        int answer_ind = obj.value("label", 0); // Правильный ответ [0-9]

        for (float &v : input) {
            v /= 255.0f;  // Приводим к диапазону [0, 1]
        }

        network.set_input_layer(input); // Установка входного слоя

        network.loop(); // Цикл активации нейронов
        error += network.network_error(answer_ind, correct); // Подсчет ошибки
        
        if (doBackProp) network.back_propagation(); // Обратное распространение ошибки

        // Прогресс
        cout << "\rProgress: " << std::round(float(line * 100) / lines_amount)
        << '%' << std::flush; 
    }

    // Процент правильных ответов и ошибка нейросети
    cout << "\nEnd of training iteration!\n";
    cout << "Correct answers: " << float(correct * 100) / lines_amount << "%\n";
    cout << "Network error: " << error << '\n';
}

int main() {
    // Объект нейросети
    Network network({784, 16, 16, 10}, 0.1f);

    int loops = 0, epoch = 0; // Кол-во эпох и текущая эпоха

    while (true) {
        cout << "How many loops: ";
        cin >> loops;

        // "Тренировка" нейросети
        for (int i = 0; i < loops; i++) {
            cout << "\n===Epoch #" << epoch << "===\n";
            train(network, "mnist_handwritten_train.json", 60000, true);
            epoch++;
        }

        // "Тест" нейросети
        cout << "\n===TEST===\n";
        train(network, "mnist_handwritten_test.json", 10000, false);

        string answer;
        cout << "\nDo you want to continue [Y/n]: ";
        cin >> answer;

        if (std::tolower(answer[0]) != 'y' && !answer.empty()) {
            break;
        }
    }
    
    // Запись файлов нейронов и смещений
    network.end_program();
}
```
## Установка
Переместите файл `network.h` в ваш проект и все =)
