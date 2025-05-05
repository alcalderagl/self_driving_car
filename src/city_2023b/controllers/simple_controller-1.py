"""camera_pid controller."""
import numpy as np
import cv2
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
from datetime import datetime
import os

# -------------------------------------------------------------------
# DEFINICIÓN DE FUNCIONES
# -------------------------------------------------------------------

# Captura y convierte la imagen de la cámara de Webots en formato utilizable por OpenCV (descartando canal alpha).
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # descartamos el canal alpha (transparencia)

# Detección de línea amarilla mediante procesamiento de imagen + HoughLines
def detectar_linea_amarilla(imagen):
    # Conversión de espacio de color BGR a HSV para facilitar la detección del color amarillo
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Rango de color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
   
    # Máscara binaria que solo mantiene los píxeles dentro del rango amarillo
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Definición de la región de interés (ROI):
    # Esto permite ignorar las zonas irrelevantes (cielo, objetos lejanos o extremos del camino)
    height, width = mask.shape
    mask[:int(height * 0.4), :] = 0                 # Oculta la parte superior (zona inútil)
    mask[:, :int(width * 0.15)] = 0                 # Oculta bordes laterales izquierdos
    mask[:, int(width * 0.85):] = 0                 # Oculta bordes laterales derechos

    # Detección de bordes con Canny + suavizado para evitar ruido
    edges = cv2.Canny(mask, 30, 100)
    blurred = cv2.GaussianBlur(edges, (5, 5), 0)

 

   # Aplicación de la Transformada de Hough para detectar líneas rectas
    lines = cv2.HoughLinesP(
        blurred,              # Imagen de entrada
        1,                    # Resolución del acumulador
        np.pi / 180,          # Resolución angular (grados convertidos a radianes)
        threshold=30,         # Cantidad mínima de intersecciones en acumulador para considerar línea
        minLineLength=20,     # Longitud mínima de línea
        maxLineGap=10         # Máxima separación entre segmentos de línea conectables
    )

    line_image = np.zeros_like(mask)  # Imagen base para dibujar líneas
    cx = None                         # Coordenada x central estimada de la línea
    x_coords = []                     # Coordenadas x de los puntos medios de líneas detectadas

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Filtro para ignorar líneas horizontales (poco útiles para seguimiento)
            if abs(y2 - y1) < 10:
                continue

            # Dibujo de la línea y cálculo del punto medio
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            x_coords.append((x1 + x2) // 2)

        # Promedio de puntos medios como estimación del centro de la línea
        if x_coords:
            cx = int(np.mean(x_coords))

    return line_image, cx  # Imagen con líneas + posición horizontal estimada

# Muestra la imagen procesada en el display de Webots
def display_image(display, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(), Display.RGB, width=image_rgb.shape[1], height=image_rgb.shape[0]
    )
    display.imagePaste(image_ref, 0, 0, False)

# -------------------------------------------------------------------
# VARIABLES GLOBALES Y CONTROL DE MOVIMIENTO
# -------------------------------------------------------------------
manual_steering = 0     # Ángulo manual acumulado
steering_angle = 0      # Ángulo actual de dirección
angle = 0.0             # Ángulo aplicado al vehículo
speed = 15              # Velocidad de crucero inicial (km/h)

def set_speed(kmh):
    global speed
    speed = kmh

# Aplica suavemente el cambio de dirección para evitar movimientos bruscos
def set_steering_angle(wheel_angle, immediate=False):
    global angle, steering_angle
    if not immediate:
        # Suavizado del cambio de ángulo para evitar giros muy agresivos
        if (wheel_angle - steering_angle) > 0.1:
            wheel_angle = steering_angle + 0.1
        elif (wheel_angle - steering_angle) < -0.1:
            wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
    wheel_angle = max(min(wheel_angle, 0.5), -0.5)  # límite físico del volante
    angle = wheel_angle

# Control manual del volante desde teclado (LEFT, RIGHT)
def change_steer_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle), turn))


# -------------------------------------------------------------------
# FUNCIÓN MAIN
# -------------------------------------------------------------------

def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")  # Ventana donde se mostrará la imagen de máscara

    keyboard = Keyboard()
    keyboard.enable(timestep)

    last_valid_angle = 0.0
    line_detected = False
    lost_line_counter = 0
    lost_line_threshold = 2  # Esperar 2 ciclos antes de confirmar pérdida de línea

    snapshot_interval = 2.0  # Tomar captura cada 2 segundos
    last_snapshot_time = -snapshot_interval

    while robot.step() != -1:
        current_time = robot.getTime()
        image = get_image(camera)

        # Detección de línea amarilla
        mask, cx = detectar_linea_amarilla(image)
        display_image(display_img, mask)

        if cx is not None:
            # Línea detectada: calcular error y corregir dirección
            width = mask.shape[1]
            error = (cx - width // 2) / (width // 2)  # Error normalizado [-1, 1]
            gain = 0.2                                # Ganancia proporcional
            current_angle = error * gain

            set_steering_angle(current_angle)
            last_valid_angle = 0.0 if abs(current_angle) < 0.2 else current_angle

            line_detected = True
            lost_line_counter = 0
            print(f"[AUTO] Línea detectada - X: {cx}, Error: {error:.2f}, Ángulo: {angle:.2f}")
        else:
            # Línea no detectada: estrategia de contingencia
            lost_line_counter += 1
            if lost_line_counter >= lost_line_threshold:
                set_steering_angle(0.0, immediate=True)
                last_valid_angle = 0.0
                print(f"[AUTO] Línea NO detectada ({lost_line_counter}) - forzando ángulo a 0.0 para seguir recto")
            else:
                print(f"[AUTO] Línea NO detectada ({lost_line_counter}) - esperando confirmación")

        # Guardar imágenes periódicamente (modo automático)
        if current_time - last_snapshot_time >= snapshot_interval:
            last_snapshot_time = current_time
            file_name = f"auto_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            camera.saveImage(os.path.join(os.getcwd(), file_name), 1)
            print(f"[AUTO] Imagen guardada: {file_name}")

        # Lectura del teclado para control manual
        key = keyboard.getKey()
        if key == keyboard.UP:
            set_speed(speed + 5.0)
        elif key == keyboard.DOWN:
            set_speed(speed - 5.0)
        elif key == keyboard.RIGHT:
            change_steer_angle(+1)
        elif key == keyboard.LEFT:
            change_steer_angle(-1)
        elif key == ord('A'):
            file_name = f"manual_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            camera.saveImage(os.path.join(os.getcwd(), file_name), 1)
            print("[MANUAL] Imagen tomada:", file_name)

        # Aplicar las decisiones al actuador del vehículo
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

# Punto de entrada del script
if __name__ == "__main__":
    main()