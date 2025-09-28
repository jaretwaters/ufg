import numpy as np 

TP_SEXO_map = {"M": "Masculino", "F": "Feminino"}

TP_COR_RACA_map = {
    0: "Não declarado/não dispõe da informação",
    1: "Branca",
    2: "Preta",
    3: "Parda",
    4: "Amarela",
    5: "Indígena",
    6: "Não declarado/não dispõe da informação",
}

TP_FAIXA_ETARIA_map = {
    1: 'Menor de 17 anos',
    2: '17 anos',
    3: '18 anos',
    4: '19 anos',
    5: '20 anos',
    6: '21 anos',
    7: '22 anos',
    8: '23 anos',
    9: '24 anos',
    10: '25 anos',
    11: 'Entre 26 e 30 anos',
    12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos',
    14: 'Entre 41 e 45 anos',
    15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos',
    17: 'Entre 56 e 60 anos',
    18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos',
    20: 'Maior de 70 anos'
}

TP_ESCOLA_map = {
    1: np.nan,
    2: "Pública",
    3: "Privada"
}

TP_LOCALIZACAO_ESC_map = {1: "Urbana", 2: "Rural"}

TP_DEPENDENCIA_ADM_ESC_map = {
    1: "Federal",
    2: "Estadual",
    3: "Municipal",
    4: "Privada"
}

TP_ENSINO_map = {
    1: 'Ensino Regular',
    2: 'Educação Especial - Modalidade Substitutiva'
}

TP_ST_CONCLUSAO_map = {
    1: 'Já concluí o Ensino Médio',
    2: 'Estou cursando e concluirei o Ensino Médio em 2023',
    3: 'Estou cursando e concluirei o Ensino Médio após 2023',
    4: 'Não concluí e não estou cursando o Ensino Médio'
}

# -----------------------
# Q001 – Q025 mappings
# -----------------------

Q001_Q002_map = {
    "A": "EF incompleto",
    "B": "EF incompleto",
    "C": "EF incompleto",
    "D": "EF completo",
    "E": "EM completo",
    "F": "Faculdade ou pós",
    "G": "Faculdade ou pós",
    "H": "Não sabe"
}

Q003_Q004_map = {
    "A": "Grupo 1: ocupações rurais / básicos",
    "B": "Grupo 2: serviços básicos, comércio",
    "C": "Grupo 3: operários / técnicos manuais",
    "D": "Grupo 4: técnicos, supervisores, pequenos empresários",
    "E": "Grupo 5: profissões de nível superior / gestores",
    "F": "Não sabe"
}

Q005_map = {
    1: "Mora sozinho",
    2: "Pequena (2–3)",
    3: "Pequena (2–3)",
    4: "Média (4–5)",
    5: "Média (4–5)",
    6: "Grande (6–8)",
    7: "Grande (6–8)",
    8: "Grande (6–8)",
    9: "Muito grande (9+)",
    10: "Muito grande (9+)",
    11: "Muito grande (9+)",
    12: "Muito grande (9+)",
    13: "Muito grande (9+)",
    14: "Muito grande (9+)",
    15: "Muito grande (9+)",
    16: "Muito grande (9+)",
    17: "Muito grande (9+)",
    18: "Muito grande (9+)",
    19: "Muito grande (9+)",
    20: "Muito grande (9+)"
}

Q006_map = {
    "A": "Nenhuma renda",
    "B": "Até 2 SM",   # até R$ 2.640
    "C": "Até 2 SM",
    "D": "Até 2 SM",
    "E": "2 a 4 SM",   # R$ 2.640 – R$ 5.280
    "F": "2 a 4 SM",
    "G": "2 a 4 SM",
    "H": "4 a 6 SM",   # R$ 5.280 – R$ 7.920
    "I": "4 a 6 SM",
    "J": "6 a 10 SM",  # R$ 7.920 – R$ 13.200
    "K": "6 a 10 SM",
    "L": "6 a 10 SM",
    "M": "6 a 10 SM",
    "N": "10 a 15 SM", # R$ 13.200 – R$ 19.800
    "O": "10 a 15 SM",
    "P": "15 a 20 SM", # R$ 19.800 – R$ 26.400
    "Q": "Acima de 20 SM"
}

# Q007 – Q025 (mostly "A = Não", "B = Sim/1", "C = Sim/2" etc.)
Q007_map = {
    "A": "Não",
    "B": "Sim",
    "C": "Sim",
    "D": "Sim"
}

Q008_Q017_map = {
    "A": "Não",
    "B": "Sim",
    "C": "Sim",
    "D": "Sim",
    "E": "Sim"
}

Q018_map = {"A": "Não", "B": "Sim"}
Q019_map = Q008_Q017_map
Q020_map = Q018_map
Q021_map = Q018_map
Q022_map = {
    "A": "Não",
    "B": "Sim",
    "C": "Sim",
    "D": "Sim",
    "E": "Sim"
}
Q023_map = Q018_map
Q024_map = Q008_Q017_map
Q025_map = Q018_map