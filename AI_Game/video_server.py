import asyncio
import base64
import cv2
import numpy as np
import websockets 
import AI_function_img_pose_add
import AI_function_img_face


async def handler(websocket):
    data = await websocket.recv()
    if data == 'face':
      while True:
          data = await websocket.recv()
          img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
          result = AI_function_img_face.handle(img)
          if result != None and len(result) > 0:
            await websocket.send(result)
    elif data == 'pose':
      while True:
          data = await websocket.recv()
          img = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
          result = AI_function_img_pose_add.handle(img)
          if result != None and len(result) > 0:
            await websocket.send(result)  


async def main():
    async with websockets.serve(handler, "172.31.11.131", 9999):
      await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())