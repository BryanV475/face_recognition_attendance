import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

def register_new_student():
    #Input the name of the student to save the image
    print("Ingrese el nombre del estudiante:")
    person_name = input()
    print("Presione \"a\"  para guardar la imagen")

    #Create the route where the image it's gonna be saved
    route = ".\\images\\{}.jpg".format(person_name)

    #Create the camera object using opencv - (0) -> First Camera
    cap = cv2.VideoCapture(0)

    #Setting the dimensions of photos taken by the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #Infinite loop to show the video captured by the camera
    while True:

        #Check if camera is grabbed in {ret: boolean} capture the image in the {frame: image}
        ret, frame = cap.read()
        cv2.imshow("Register", frame)
        
        #If the camera is not grabbed correctly breaks the loop
        if not ret:
            break
        
        #Listens for a keyboard input
        key = cv2.waitKey(1)

        #If the keyboard input -> "a"
        if (key%256) == 97:

            #Saving the image {frame} in the created route
            cv2.imwrite(route,frame)

            #Breaking the loop
            break

    #Release and close the camera object  
    cap.release()
    cv2.destroyAllWindows()

def get_images(path):
    #Array for images
    images = []

    #Array for persons in images names
    names = []

    #List all images in the directory
    directory = os.listdir(path)

    for image in directory:
        #Read the image from the directory
        readimg = cv2.imread(f'{path}/{image}')

        #Append image to images array
        images.append(readimg)

        #Images names
        names.append(os.path.splitext(image)[0])
    
    #Return images and names
    return images, names

def face_codification(images):
    #Array for codificated faces
    codificated = []

    for image in images:
        
        #Color correction from BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Codification of the image using face_recognition as fr
        codification = fr.face_encodings(image)[0]

        #Append the codification to out <codificated> list
        codificated.append(codification)

    #Return codificated faces array
    return codificated

def write_attendance(name):
    #Open the .csv in read-write mode
    with open('attendance.csv','r+') as file:
        #Read the information in the file
        data = file.readline()

        #Create the names list
        names_in_file = []

        #Iterate in each line of the file
        for line in data:
            #Search for input and split it with < , > -> input= [ User<Name>, Date<AAAA,MM,DD>, Hour<HH.MM.ss> ]
            input = line.split(',')

            #Save the names
            names_in_file.append(input[0])
        
        #Check if we don't have the name in the file to create a new record
        if name not in names_in_file:
            #Get the date and time
            date_info = datetime.now()

            #Extract the Date<AAAA,MM,DD>
            date = date_info.strftime('%Y:%m:%d')

            #Extract the hour
            hour = date_info.strftime('%H:%M:%S')

            #Save the new record [ User<Name>, Date<AAAA,MM,DD>, Hour<HH.MM.ss> ]
            file.writelines(f'\n{name},{date},{hour}')

def get_attendance(images, names):
    #Codificate our faces
    codificated_faces = face_codification(images)

    #Create the list for users encountered
    encountered = []

    #Set a control variable for the faces comparison
    comp1 = 100

    #Create the camera object using opencv - (0) -> First Camera
    cap = cv2.VideoCapture(0)

    #Setting the dimensions of the camera
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #Infinite loop to process camera in real time
    while True:

        #Read the frames
        ret, frame = cap.read()

        #Reduce the images to improve efficiency
        frame2 = cv2.resize(frame, (0,0), None, 0.25, 0.25)

        #Convert BGR -> RGB color
        rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        #Search for faces in the camera's vision field
        faces = fr.face_locations(rgb)
        
        #Codificate the faces located
        facescod = fr.face_encodings(rgb, faces)

        #Iterate in the faces encountered and each one's location facecod<codification>, faceloc<location>
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
                name = names[minimum].upper()

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
                        write_attendance(name)


        #Show the video recording in real time
        cv2.imshow('Attendance Recognition', frame)

        #Listens for a keyboard input
        key = cv2.waitKey(1)

        #If the keyboard input -> "a"
        if (key%256) == 97:

            #Breaking the loop
            break

    #Release and close the camera object  
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    #Display a options menu using an infinite loop
    while True:
        #Initialize the option variable
        option = 0
        
        #Clear the console
        os.system('cls')

        #Print the options
        print("\tSistema de asistencia")
        print("1. Registrar nuevo estudiante")
        print("2. Registrar asistencia")
        print("0. Salir")
        
        #Get the option
        option = input('Respuesta: ')

        #Exit condition
        if option != "1" and option != "2" :
            break
        
        #Converts the option string into a number
        option = int(option)

        #Control the options
        if option == 1:
            register_new_student()

        if option == 2:
            images, names = get_images('images')
            get_attendance(images,names)
