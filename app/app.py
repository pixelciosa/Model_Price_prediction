import gradio as gr
import pandas as pd
import pickle


# Parámetros
PARAMS_NAME = [
    "rooms",
    "bathrooms",
    "surface_covered",
    "neighborhood",
    "bathrooms_missing",
    "rooms_missing"
]

# LabelEncoder
ENCODER_PATH = "models/le_neighborhood.pkl"
with open(ENCODER_PATH, 'rb') as handle:
   le = pickle.load(handle)


# Scaler
SCALER_PATH = "models/scalers.pkl"
with open(SCALER_PATH, 'rb') as handle:
    scalers = pickle.load(handle)

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
    
    # Convierto el string de neighborhood con LabelEncoder
    single_instance['neighborhood'] = le.transform(single_instance['neighborhood'])

    # Transformar variables numéricas con Scaler manteniendo el DataFrame
    surface_real = single_instance['surface_covered'].values[0] # guardo el dato sin transformar para usarlo despues
    num_cols = ['rooms', 'bathrooms', 'surface_covered']
    for col in num_cols:
        single_instance[col] = scalers[col].transform(single_instance[[col]])

    # Seleccionar todas las columnas que espera el modelo en el mismo orden
    features = ['neighborhood', 'rooms', 'bathrooms', 'surface_covered', 'bathrooms_missing', 'rooms_missing']

    # Realizo la predicción
    prediction_USD_per_m2 = model.predict(single_instance[features])
    
    # Multiplicar por superficie real
    total_USD = prediction_USD_per_m2 * surface_real

    response = format(total_USD[0], '.0f')

    return ('USD $'+response)

def update_slider_interactive(checkbox_value):
    """
    Si el checkbox es True, el slider es NO interactivo.
    Si el checkbox es False, el slider es interactivo.
    """
    # El valor del checkbox es el mismo valor que queremos para 'interactive'
    return gr.Slider(visible= not checkbox_value)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center'>
        Valuación de Alquileres de Oficinas en CABA 🏢
        </h1>
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
            surface_covered = gr.Slider(
                label="Superficie cubierta en metros cuadrados",
                minimum=1, maximum=10000,
                step=1,
                value=100
                )
            rooms_missing = gr.Checkbox(
                label="Cantidad de ambientes sin determinar",
                value=True)
            rooms = gr.Slider(
                label="Cantidad de Ambientes",
                minimum=1, maximum=100,
                step=1,
                value=2,
                visible=not rooms_missing.value
                )
            bathrooms_missing = gr.Checkbox(
                label="Cantidad de baños sin determinar",
                value=False)
            bathrooms= gr.Dropdown(
                label="Cantidad de baños",
                choices=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10 ],
                value=1,
                visible=not bathrooms_missing.value
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
       'Villa Devoto', 'Villa Urquiza', 'Villa del Parque'],
                value='Abasto',
                )
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
                    surface_covered,
                    neighborhood,
                    bathrooms_missing,
                    rooms_missing
                ],
                outputs=[label],
                api_name="prediccion"
            )

            
        with gr.Column():
            gr.Image(
                value="images/Mapa-CABA.svg",
                show_label=False
                )
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto creado por Adriana Villalobos en el bootcamp de Data Science y MLops de EDVai 🤓
            </a>
        </p>
        """
    )

    # Interacción: Cuando el checkbox cambia, llama a la función
    rooms_missing.change(
        fn=update_slider_interactive, # La función con la lógica 
        inputs=[rooms_missing],       # Lee el estado del checkbox
        outputs=[rooms]               # Actualiza el slider
    )
    bathrooms_missing.change(
        fn=update_slider_interactive,     # La función con la lógica 
        inputs=[bathrooms_missing],       # Lee el estado del checkbox
        outputs=[bathrooms]               # Actualiza el slider
    )

demo.launch()
