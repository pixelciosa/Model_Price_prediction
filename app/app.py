import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


# Parámetros
PARAMS_NAME = [
    "rooms",
    "bathrooms",
    "surface_total",
    "neighborhood"
]

# Columnas
COLUMNS_PATH = "models/columns_labelEncoder.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
   le = pickle.load(handle)

# Modelo
MODEL_PATH = "models/model.pkl"
with open(MODEL_PATH, 'rb') as handle:
    model = pickle.load(handle)

# Realizar predicción
def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    # Reformat neighborhood column
    le = LabelEncoder()
    single_instance['neighborhood'] = le.fit_transform(single_instance['neighborhood'])
    print(single_instance)
    prediction_USD_per_m2 = model.predict(single_instance)
    total_USD = prediction_USD_per_m2 * single_instance['surface_total'].values[0]

    response = format(total_USD[0], '.0f')
    #print(response)
    return (response)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Valuación de Alquileres de Oficinas en CABA 🏢
        """
    )

    with gr.Row():
        gr.Markdown(
                """
                ## Características de la oficina
                """
            )
    with gr.Row():
        with gr.Column():
            rooms = gr.Slider(
                label="Cantidad de Ambientes",
                minimum=1, maximum=100,
                step=1,
                value=3
                )
            bathrooms= gr.Dropdown(
                label="Cantidad de baños",
                choices=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10 ],
                value=1
                )
            surface = gr.Slider(
                label="Superficie cubierta en metros cuadrados",
                minimum=1, maximum=10000,
                step=1,
                value=100
                )
            neighborhood = gr.Dropdown(
                label="Barrio",
                choices=['Abasto', 'Almagro', 'Balvanera', 'Barracas', 'Barrio Norte',
       'Belgrano', 'Boca', 'Boedo', 'Caballito', 'Catalinas',
       'Centro / Microcentro', 'Chacarita', 'Colegiales', 'Congreso',
       'Constitución', 'Flores', 'Floresta', 'Liniers', 'Mataderos',
       'Monserrat', 'Nuñez', 'Once', 'Otros', 'Palermo',
       'Parque Chacabuco', 'Parque Patricios', 'Paternal',
       'Puerto Madero', 'Recoleta', 'Retiro', 'Saavedra', 'San Cristobal',
       'San Nicolás', 'San Telmo', 'Tribunales', 'Villa Crespo',
       'Villa Devoto', 'Villa Urquiza', 'Villa del Parque', 'Otro'],
                value='Abasto',
                )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## Predicción en dólares mensuales
                """
            )

            label = gr.Label(label="Valor aproximado del alquiler")
            predict_btn = gr.Button(value="Calcular")
            predict_btn.click(
                predict,
                inputs=[
                    rooms,
                    bathrooms,
                    surface,
                    neighborhood
                ],
                outputs=[label],
                api_name="prediccion"
            )
        with gr.Column():
            gr.Image(
                value="app/images/Mapa-CABA.svg",
                show_label=False
                )
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto creado por Adriana Villalobos en el bootcamp de EDVAI 🤗
            </a>
        </p>
        """
    )

demo.launch(share = True)
