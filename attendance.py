import cv2
import os
import numpy as np
import face_recognition as fr
import random
from datetime import datetime
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QInputDialog, QPushButton, QHBoxLayout, QVBoxLayout, QDialog, QMainWindow, QTableWidget, QTableWidgetItem


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Sistema de Gestión de Asistencias")
        self.setFixedSize(500, 300)
        font = QFont()
        font.setPointSize(12) # set font size
        font.setBold(True) 

        # Create UI elements
        self.label = QLabel("Bienvenido al Sistema de Gestión de Asistencias")
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setFont(font)

        # Add image to the UI
        self.image_label = QLabel(self)
        pixmap = QPixmap('resources/logo.png')
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.register_button = QPushButton("Registrar nuevo estudiante")
        self.register_button.clicked.connect(self.register_student)

        self.attendance_button = QPushButton("Registrar Asistencia")
        self.attendance_button.clicked.connect(self.register_attendance)

        self.close_button = QPushButton("Salir")
        self.close_button.clicked.connect(self.close)

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.image_label)
        vbox.addStretch(1)
        vbox.addWidget(self.register_button)
        vbox.addWidget(self.attendance_button)
        vbox.addWidget(self.close_button)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)
        hbox.addStretch(1)

        self.setLayout(hbox)

    def register_student(self):
        # Show TextInput and accept button
        name, ok = QInputDialog.getText(self, "Nuevo", "Ingrese el nombre del estudiante:")
        if ok and name:
            route = ".\\images\\{}.jpg".format(name)
            cap = cv2.VideoCapture(0)
            #Setting the dimensions of photos taken by the camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("Nuevo estudiante registrado correctamente! Bienvenido ", name)
            image_saved = False
            try: 
                while True:
                    ret, frame = cap.read()
                    frame_copy = frame.copy()  # create a copy of the original frame
                    cv2.putText(frame_copy, "Presione 'a' para guardar la imagen o 'q' para salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Registrar Estudiante", frame_copy)
                    if not ret:
                        break
                    key = cv2.waitKey(1)
                    if (key & 0xFF == ord('q')) or (key % 256 == 97):
                        if key % 256 == 97:
                            #Saving the image {frame} in the created route
                            cv2.imwrite(route, frame)
                            break
                cap.release()
                cv2.destroyAllWindows()
            except:
                print("Ocurrio un error al capturar la imagen.")
                cap.release()
                cv2.destroyAllWindows()

    def register_attendance(self):

        images = []
        names = []
        #-- Read Images --#
        path = 'images'
        images = [cv2.imread(f'{path}/{image}') for image in os.listdir(path)]
        names = [os.path.splitext(image)[0] for image in os.listdir(path)]
        
        if len(images) != 0:
            # Codificate our faces
            codificated_faces = [fr.face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0] for image in images]
            # Create the list for users encountered
            encountered = []
            # Set a control variable for the faces comparison
            comp1 = 100
            # Create the camera object using opencv - (0) -> First Camera
            cap = cv2.VideoCapture(0)
            while True:
                # Read the frames
                ret, frame = cap.read()
                # Reduce the images to improve efficiency
                frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)
                # Convert BGR -> RGB color
                rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                # Search for faces in the camera's vision field
                faces = fr.face_locations(rgb)
                # Codificate the faces located
                facescod = fr.face_encodings(rgb, faces)

                # Iterate in the faces encountered and each one's location facecod<codification>, faceloc<location>
                for facecod, faceloc in zip(facescod, faces):
                    #Comparison between registered faces and encountered faces in real time
                    comparison = fr.compare_faces(codificated_faces, facecod)

                    #Calculate the similarity -> similarity = [distance_to_face(1), distance_to_face(2), ... distance_to_face(n)]
                    #distance_to_face(n)<Double> -> is lower according to the similarity to the registered face
                    similarity = fr.face_distance(codificated_faces, facecod)

                    #We search for the minimum value index
                    minimum = np.argmin(similarity)

                    #Check if the minimum value is in our comparison array
                    if comparison[minimum]:
                        #We get the name of the user
                        if similarity[minimum] < 0.5:
                            name = names[minimum].upper()
                        else:
                            name = "Desconocido"

                        #Extract the coordinates of the face
                        yi, xf, yf, xi = faceloc

                        #Resize the coordinates
                        yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4

                        #Get the index of the location
                        index = comparison.index(True)

                        #Compare if the index of the face location to set up a color for the rectangle around the face
                        if comp1 != index:
                            #Set colors to draw the rectangle
                            r = random.randrange(0, 255, 50)
                            g = random.randrange(0, 255, 50)
                            b = random.randrange(0, 255, 50)

                            #Set the control variable to draw the rectangle
                            comp1 = index
                        
                        if comp1 == index:
                            #Draw the rectangle from (xi, yi) to (xf, yf) with color (r, g, b) and a thickness of 3
                            cv2.rectangle(frame,(xi, yi), (xf, yf), (r, g, b), 3)
                            
                            #Draw a filled rectangle to display the name of the user
                            cv2.rectangle(frame,(xi, yf-35), (xf, yf), (r, g, b), cv2.FILLED)

                            #Display the name in the filled rectangle
                            cv2.putText(frame, name, (xi+6, yf-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            #Save the record of the user attendance if is not recorded already
                            if name not in encountered:
                                encountered.append(name)
                                #Save the record
                                #Open the .csv in read-write mode
                                if name != "Desconocido":
                                    with open('attendance.csv','r+') as file:
                                        #Read the information in the file
                                        data = file.readline()
                                        #Get the date and time
                                        date_info = datetime.now()
                                        #Extract the Date<AAAA,MM,DD>
                                        date = date_info.strftime('%Y:%m:%d')
                                        #Extract the hour
                                        hour = date_info.strftime('%H:%M:%S')
                                        #Save the new record [ User<Name>, Date<AAAA,MM,DD>, Hour<HH.MM.ss> ]
                                        file.writelines(f'\n{name},{date},{hour}')
                            
                    else:
                        # If the face is unknown, draw a red rectangle and display "Desconocido"
                        yi, xf, yf, xi = faceloc
                        yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4
                        cv2.rectangle(frame,(xi, yi), (xf, yf), (0,0,255), 3)
                        cv2.rectangle(frame,(xi, yf-35), (xf, yf), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, "Desconocido", (xi+6, yf-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



                #Show the video recording in real time
                cv2.imshow('Registro de Asistencia', frame)

                #Listens for a keyboard input
                key = cv2.waitKey(1)

                #If the keyboard input -> "a"
                if (key%256) == 97:
                    #Breaking the loop
                    break

            #Release and close the camera object  
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("No hay registros")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
