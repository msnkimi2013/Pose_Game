from simple_websocket_server import WebSocketServer, WebSocket
import base64
import cv2
import numpy as np
import warnings





warnings.simplefilter("ignore", DeprecationWarning)





class SimpleEcho(WebSocket):
    print("connect_suc")
    
    def handle(self):

        msg = self.data
        print(msg)
        # video = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        


        # Jin code here ***



        # output label text ***
        # answer = ai_model(video)



        # cv2.imshow('image', video)
        # cv2.waitKey(1)

        # self.send_message(answer)





    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')





server = WebSocketServer('172.31.11.131', 9999, SimpleEcho)
server.serve_forever()

