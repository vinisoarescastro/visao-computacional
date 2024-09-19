import cv2 

# carrega a imagem a parti do caminho e armazena na variavel 'img'
img = cv2.imread(r'C:\Users\vinis\OneDrive\Documents\Programacao\visao-computacional\deteccao-de-faces\img\foto-inf4.jpeg')

# verificar as dimensoes da imagem carregada e imprime no terminal
print(img.shape)

# verifica se a imagem foi carregada corretamente 
if img is not None:
    print('Imagem carregada com sucesso.')
    cv2.imshow('Imagem', img)   # mostra a imagem original em um janela
    cv2.waitKey(0)              # aguarda qualquer tecla ser pressionada pra continuar
    cv2.destroyAllWindows()     # fecha todas as janelas aberta do opencv
else:
    print('Erro ao carregar a imagem.') # exibe mensagem de erro se a imagem não for carregada.

img2 = cv2.resize(img, (1280, 960))  # redimensiona a imagem para 1280x960 px e armazena a nova img na variavel img2. 
print(img2.shape)                   # verifica as dimensoes e mostra no terminal

img2_cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #converte a imagem dimensionada para escala de cinzas

# verifica se a conversão para escala de cinzas foi realizada com sucesso. 
if img2_cinza is not None:
    print('Imagem em escala de cinzas carregada com sucesso.')
    cv2.imshow('Imagem', img2_cinza)    # mostra a imagem oem escala de cinzas em um janela
    cv2.waitKey(0)                      # aguarda qualquer tecla ser pressionada pra continuar
    cv2.destroyAllWindows()             # fecha todas as janelas aberta do opencv
else:
    print('Erro ao carregar imagem em escala de cinza.')    # exibe mensagem de erro se a imagem não for carregada.

print(img2_cinza.shape) # verificar as dimensoes da imagem carregada e imprime no terminal

# carrega o classificador em cascata para detecção de faces
detector_facial = cv2.CascadeClassifier(r'C:\Users\vinis\OneDrive\Documents\Programacao\visao-computacional\deteccao-de-faces\cascades\haarcascade_frontalface_default.xml')

deteccoes = detector_facial.detectMultiScale(img2_cinza, scaleFactor=1.39)  
# detecta faces na imagem em escala de cinzas. `scaleFactor=1.08` ajusta como o detector faz a busca.
print(deteccoes)        # mostra as coordenadas das deteccções
print(len(deteccoes))   # mostra o número de faces detectadas


# desenha retangulos ao redor das faces detectadas na imagem redimensionada 
for x, y, w, h in deteccoes:
    cv2.rectangle(img2, (x,y), (x+w, y+h), (0,255,52), 2)
    # desenha um retangulo com a cor (0,255,52) e expressura de 2 pixels


cv2.imshow('Imagem com detecção de faces', img2)    # exibe a imagem com retângulos ao redor dasfaces detectadas
cv2.waitKey(0)                                      # aguarda qualquer tecla ser pressionada pra continuar
cv2.destroyAllWindows()                             # fecha todas as janelas aberta do opencv