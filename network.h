#pragma once

#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <random>
#include <vector>

using json = nlohmann::json;
using std::cout;
using std::cin;
using std::string;
using std::vector;


// Класс нейрона содержит переменные:
// [value]   - выходное значение
// [sum]     - взвешенная сумма
// [bias]    - смещение
// [error]   - ошибка нейрона
// [weights] - веса, входящие в нейрон
// А также методы для вычисления этих значений
class Neuron {
public:
    float value, sum, bias, error;
    vector<float> weights;

    Neuron(float bias, vector<float> weights) : 
        bias(bias), weights(weights), value(0), sum(0), error(0) {}

    // Функция активации - вычисляет вз. сумму и выходное значение
    void activate(vector<float> &neurons) {
        sum = 0;
        for (int i = 0; i < neurons.size(); i++) {
            sum += neurons.at(i) * weights.at(i);
        }
        sum += bias;

        value = sigmoid(sum);
    }

    void activate(vector<Neuron> &neurons) {
        sum = 0;
        for (int i = 0; i < neurons.size(); i++) {
            sum += neurons.at(i).value * weights.at(i);
        }
        sum += bias;

        value = sigmoid(sum);
    }

    // Подсчет ошибки на внешнем слое
    void count_error(const float &answer) {
        error = (value - answer) * sigmoid_der(value);
    }

    // Подсчет ошибки на скрытом слое
    void count_error(const vector<Neuron> &layer, int &ind) {
        error = 0;

        for (const Neuron &n : layer) {
            error += n.error * n.weights.at(ind);
        }

        error *= sigmoid_der(value);
    }

    // Градиентный спуск
    void gradient_descent(const vector<Neuron> &neurons, const float &learn_val) {
        // Цикл, где w - индекс веса, связывающего предыдущие нейроны с текущим
        for (int w = 0; w < weights.size(); w++) {
            // Градиентный спуск по весам
            weights.at(w) -= neurons.at(w).value * error * learn_val;
        }
        // Градиентный спуск по смещениям
        bias -= error * learn_val;
    }

    void gradient_descent(const vector<float> &neurons, const float &learn_val) {
        for (int w = 0; w < weights.size(); w++) {
            weights.at(w) -= neurons.at(w) * error * learn_val;
        }
        bias -= error * learn_val;
    }

    float sigmoid(const float &x) {
        return 1.0f / (1.0 + std::exp(-x));
    }

    float sigmoid_der(const float &x) {
        return x * (1.0 - x);
    }
};

// Основной класс нейросети
class Network {
    // Скорость обучения нейросети
    const float learning_val;

    // Входной и остальные слои
    vector<float> input_layer;
    vector<vector<Neuron>> layers;

    void generateFiles(vector<int> &neurons_amount) {
        // Файлы - "weights.json", "biases.json"
        std::ofstream file_w("weights.json");
        std::ofstream file_b("biases.json");
    
        if (file_w.is_open() && file_b.is_open()) {
            // Переменные для создания случайных значений от -0.5 до 0.5
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> distr(-0.5, 0.5);

            // ОбЪекты файлов
            json weights, biases;

            // Цикл где i - номер слоя нейросети
            for (int i = 1; i < neurons_amount.size(); i++) {
                // Векторы весов и смещений всех нейронов i слоя
                vector<vector<float>> matrix_w(neurons_amount.at(i));
                vector<float> matrix_b(neurons_amount.at(i));

                // Цикл где x - индекс нейрона нейросети
                for (int x = 0; x < neurons_amount.at(i); x++) {
                    // "Бронирование" весов для x нейрона i слоя
                    matrix_w.at(x).reserve(neurons_amount.at(i - 1));

                    // Случайное значение смещения для x нейрона
                    matrix_b.at(x) = distr(gen);

                    // Цикл где z - индекс веса, связанного с нейроном
                    for (int z = 0; z < neurons_amount.at(i - 1); z++) {
                        // Случайное значение z веса
                        matrix_w.at(x).push_back(distr(gen));
                    }
                }

                // Запись в объекты файлов в формате: "layer1" и тд.
                weights["layer" + std::to_string(i)] = matrix_w;
                biases["layer" + std::to_string(i)] = matrix_b;
            }

            // Запись объектов в файлы
            file_w << std::setw(4) << weights << std::endl;
            file_b << std::setw(4) << biases << std::endl;

            file_w.close();
            file_b.close();

            cout << "Files generated!\n";
        }
    }

    void init_Neurons(vector<int> &neurons_amount) {
        // Файлы "weights.json", "biases.json" в режиме чтения
        std::ifstream file_w("weights.json");
        std::ifstream file_b("biases.json");

        if (file_w.is_open() && file_b.is_open()) {
            // Объекты файлов
            json weights, biases;
            file_w >> weights;
            file_b >> biases;

            // Установка всех значений входного слоя в 0
            for (float &v : input_layer) {
                v = 0;
            }

            // Цикл где l - номер слоя нейросети
            for (int l = 1; l < layers.size() + 1; l++) {
                // "Бронирование" нейронов для l слоя
                layers.at(l - 1).reserve(neurons_amount.at(l));

                // Цикл где i - индекс нейрона l слоя
                for (int i = 0; i < neurons_amount.at(l); i++) {
                    // Инициализация нейрона со значениями из файла
                    layers.at(l - 1).push_back(
                        Neuron(get_bias(biases, l, i), get_weights(weights, l, i)));
                }
            }
            cout << "Weights and biases initialized!\n";
        }
    }

    vector<float> get_weights(json &weights, const int &layer, const int &ind) {
        return weights["layer" + std::to_string(layer)][ind];
    }

    float get_bias(json &biases, const int &layer, const int &ind) {
        return biases["layer" + std::to_string(layer)][ind];
    }

    void writeFiles() {
        std::ofstream file_w("weights.json");
        std::ofstream file_b("biases.json");

        if (file_w.is_open() && file_b.is_open()) {
            // ОбЪекты файлов
            json weights, biases;

            // Цикл где l - индекс слоя
            for (int l = 0; l < layers.size(); l++) {

                // Векторы весов и смещений
                vector<vector<float>> matrix_w(layers.at(l).size());
                vector<float> matrix_b(layers.at(l).size());

                // Цикл где n - индекс нейрона l слоя
                for (int n = 0; n < layers.at(l).size(); n++) {

                    // Значения весов и смещений для n нейрона
                    matrix_b.at(n) = layers.at(l).at(n).bias;
                    matrix_w.at(n) = layers.at(l).at(n).weights;
                }

                // Запись в объекты файлов в формате: "layer1" и тд.
                weights["layer" + std::to_string(l + 1)] = matrix_w;
                biases["layer" + std::to_string(l + 1)] = matrix_b;
            }

            // Запись объектов в файлы
            file_w << std::setw(4) << weights << std::endl;
            file_b << std::setw(4) << biases << std::endl;

            file_w.close();
            file_b.close();

            cout << "Files written!\n";
        }
    }

public:
    // Инициализация
    // Принимает список нейронов в нейросети в формате: {64, 16, 8, ...}
    // То есть на 1 слое 64 нейрона, на 2 - 16 нейронов, на 3 - 8 и тд.
    Network(vector<int> neurons_amount, float learning_val) : 
        input_layer(neurons_amount.at(0)), layers(neurons_amount.size() - 1), 
        learning_val(learning_val) {
        
        // Проверка нужно ли создать файлы весов и смещений
        string answer;
        cout << "Do you want to generate files [Y/n]: ";
        cin >> answer;

        if (std::tolower(answer[0]) == 'y' || answer.empty()) {
            // Создание файлов
            generateFiles(neurons_amount);
        }
        
        // Инициализация нейронов
        init_Neurons(neurons_amount);
    }

    void end_program() {
        // Проверка нужно ли записать файлы весов и смещений
        string answer;
        cout << "Do you want to write files [Y/n]: ";
        cin >> answer;

        if (std::tolower(answer[0]) == 'y' || answer.empty()) {
            // Запись файлов
            writeFiles();
        }
    }

    void set_input_layer(vector<float> &values) {
        input_layer = values;
    }

    void loop() {
        // Активация нейронов 1 слоя
        for (Neuron &n : layers.at(0)) {
            n.activate(input_layer);
        }
        // Активация нейронов 2, 3, ... слоя
        for (int i = 1; i < layers.size(); i++) {
            for (Neuron &n : layers.at(i)) {
                n.activate(layers.at(i - 1));
            }
        }
    }

    float network_error(const int &answer_ind, int &correct) {
        // Переменные ошибки нейросети и правильного ответа
        float error = 0, y = 0;
        int answer = 0;

        // Цикл, где i - индекс нейрона последнего слоя
        for (int i = 0; i < layers.back().size(); i++) {
            // Вычисление ожидаемого ответа
            y = (i == answer_ind) ? 1 : 0;

            // Вычисление ошибки нейрона
            error += std::pow(layers.back().at(i).value - y, 2) / 2;
            layers.back().at(i).count_error(y);

            // Поиск индекса максимального значения нейрона
            if (layers.back().at(i).value > layers.back().at(answer).value) {
                answer = i;
            }
        }

        // Если индекс совпадает с правильным добавляем 1
        if (answer == answer_ind) {
            correct++;
        }

        // Среднее ареф. от ошибок нейронов
        return error / layers.back().size();
    }

    void back_propagation() {
        // Цикл, где l - индекс невходного слоя нейросети
        for (int l = layers.size() - 2; l >= 0; l--) {
            // Подсчет ошибок нейронов l слоя
            for (int n = 0; n < layers.at(l).size(); n++) {
                layers.at(l).at(n).count_error(layers.at(l + 1), n);
            }

            // Градиентный спуск весов и смещений l + 1 слоя
            for (Neuron &n : layers.at(l + 1)) {
                n.gradient_descent(layers.at(l), learning_val);
            }
        }
        
        // Градиентный спуск 1 слоя нейросети
        for (Neuron &n : layers.at(0)) {
            n.gradient_descent(input_layer, learning_val);
        }
    }
};