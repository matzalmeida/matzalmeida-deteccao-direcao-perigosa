# matzalmeida-deteccao-direcao-perigosa

## Resumo

A detecção de direção perigosa é fundamental para melhorar a segurança no trânsito e prevenir acidentes causados por comportamentos arriscados. Este trabalho propõe um sis- tema de classificação de direção perigosa baseado em aprendizado de máquina, utilizando dados sensoriais inerciais capturados por um automodelo em ambiente simulado. A coleta de dados incluiu registros de acelerômetros e giroscópios para identificar padrões associados a direção perigosa e segura, permitindo a execução de manobras que seriam inviáveis em condições reais. Três modelos de aprendizado de máquina foram implementados e comparados: SVM, FNN e LSTM. As técnicas de pré-processamento, como decimação com filtros e por média móvel, foram aplicadas para otimizar os dados. Os resultados indicaram que o modelo LSTM alcançou o melhor desempenho devido à sua capacidade de lidar com sequências temporais, seguido pelo SVM, que apresentou elevada precisão e recall com dados filtrados. O modelo FNN demonstrou sensibilidade ao pré-processamento, apresentando desempenho inferior sem técnicas de filtragem. Conclui-se que os modelos avaliados apresentam grande potencial para aplicações práticas, como sistemas de alerta em tempo real, contribuindo para a redução de acidentes e promovendo um trânsito mais seguro. A utilização de dados simulados oferece uma base promissora para replicar resultados em cenários reais, embora futuras pesquisas devam abordar os desafios associados à coleta e processamento de dados no mundo real.

## Objetivo geral

Este trabalho busca coletar dados sensoriais inerciais através de um auto-modelo inserido em um ambiente simulado para a prototipação e o desenvolvimento de soluções de aprendizado de máquina capazes de classificar um trecho como perigoso ou seguro que pode posteriormente ser utilizado por um sistema de alerta para motoristas.

## Objetivos específicos

- Realizar a coleta de dados usando o auto-modelo;
- Realizar o ajuste dos dados para o formato adequado a ser treinado pelos modelos de aprendizado de máquina;
- Implementar e aplicar os algoritmos de aprendizado de máquina selecionados nos dados tratados;
- Avaliar a acurácia e a precisão dos algoritmos bem como o comparativos entre os modelos testados, além de outras métricas relacionadas ao desempenho.

---
