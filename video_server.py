import asyncio
import base64
import cv2
import numpy as np
import websockets 
import AI_function_img
# import AI_function_img_face


async def handler(websocket):
    while True:
        data = await websocket.recv()
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        


        result = AI_function_img.handle(img)       # jin
        # result = AI_function_img_face.handle(img)       # jin
        


        if result != None and len(result) > 0:
          await websocket.send(result)


async def main():
    async with websockets.serve(handler, "172.31.11.131", 9999):
      await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())