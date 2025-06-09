import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
from PIL import Image
import io
import os
import uuid # Import the uuid module
import asyncio
import chromadb
import re # Import the re module for regular expressions
import traceback # Import the traceback module
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Form
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Define the path to the .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")

# Load environment variables from the .env file
load_dotenv(dotenv_path=dotenv_path)

# Get API key from environment variables
API_KEY = os.getenv("API_KEY")

# If API_KEY is not set, generate one and save it to the .env file
if not API_KEY:
    API_KEY = str(uuid.uuid4())
    print(f"API_KEY not found. Generating a new one: {API_KEY}")
    try:
        # Ensure the .env file exists before trying to set a key
        if not os.path.exists(dotenv_path):
            with open(dotenv_path, 'w') as f:
                f.write("") # Create an empty .env file if it doesn't exist
        set_key(dotenv_path, "API_KEY", API_KEY)
        print(f"API_KEY saved to {dotenv_path}")
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Failed to set API_KEY: {e}")

# Reload environment variables to ensure the newly generated API_KEY is loaded
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("API_KEY") # Re-read the API_KEY after potentially writing it

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

# Thiết bị: Sử dụng GPU nếu có, ngược lại sử dụng CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Đang sử dụng thiết bị: {device}')

# --- 1. Khởi tạo các mô hình ---
# MTCNN để phát hiện và căn chỉnh khuôn mặt
mtcnn = MTCNN(
    image_size=160, # Kích thước khuôn mặt đầu ra mong muốn
    margin=0,       # Biên độ quanh khuôn mặt
    min_face_size=20, # Kích thước khuôn mặt tối thiểu để phát hiện
    thresholds=[0.6, 0.7, 0.7], # Ngưỡng xác suất cho các bước của MTCNN
    factor=0.709,   # Tham số tỷ lệ
    post_process=True, # Xử lý hậu kỳ (làm mịn bounding box)
    device=device,   # Chạy trên thiết bị đã chọn,
    keep_all=True # Giữ lại toàn bộ khuôn mặt
)

# Inception Resnet V1 (FaceNet) để tạo embeddings khuôn mặt
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # Sử dụng weights từ tập VGGFace2

print("Đã khởi tạo mô hình MTCNN và FaceNet.")

# --- 2. Xây dựng Cơ sở dữ liệu Vector (Enrollment) với ChromaDB ---

# Khởi tạo ChromaDB client
# Để lưu vào ổ đĩa, bạn cần truyền path vào PersistentClient
client = chromadb.PersistentClient(path="./chromadb_data")

# Dictionary để lưu trữ các collection đã tải vào bộ nhớ
# Key: collection_name, Value: dictionary {person_name: embedding_tensor}
loaded_collections = {}

def get_or_create_chroma_collection(collection_name: str):
    """Tạo hoặc lấy collection ChromaDB."""
    # Sanitize the collection name to adhere to ChromaDB's naming rules
    # Replace spaces and other invalid characters with underscores, and convert to lowercase
    sanitized_collection_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name).lower()
    # Ensure it starts and ends with an alphanumeric character if it doesn't already
    sanitized_collection_name = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized_collection_name)
    sanitized_collection_name = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized_collection_name)

    # Ensure the name is not empty after sanitization and meets minimum length
    if not sanitized_collection_name:
        raise ValueError("Collection name cannot be empty after sanitization.")
    if len(sanitized_collection_name) < 3:
        sanitized_collection_name = (sanitized_collection_name + "___")[:3] # Pad to min length if too short

    try:
        collection = client.get_or_create_collection(name=sanitized_collection_name)
        print(f"Đã kết nối tới ChromaDB collection: {sanitized_collection_name}")
        return collection
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi kết nối hoặc tạo ChromaDB collection: {e}. Tên collection đã thử: {sanitized_collection_name}")

def load_database_to_memory(collection_name: str):
    """Tải dữ liệu từ ChromaDB collection vào bộ nhớ dưới dạng dictionary."""
    collection = get_or_create_chroma_collection(collection_name)
    try:
        results = collection.get(
            ids=collection.get()['ids'], # Lấy tất cả các ID
            include=['embeddings', 'metadatas']
        )
        database_dict = {}
        for i in range(len(results['ids'])):
            person_name = results['ids'][i]
            embedding_np = np.array(results['embeddings'][i]).astype(np.float32)
            embedding_tensor = torch.from_numpy(embedding_np).to(device)
            database_dict[person_name] = embedding_tensor
        print(f"Đã tải {len(database_dict)} bản ghi từ ChromaDB collection '{collection_name}'.")
        loaded_collections[collection_name] = database_dict
        return database_dict
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        print(f"Lỗi khi tải dữ liệu từ ChromaDB collection '{collection_name}': {e}")
        return {}

# Hàm để thêm một người vào database ChromaDB
# Chuyển hàm này thành đồng bộ để có thể chạy trong executor
def add_person_to_database_sync(image_bytes: bytes, person_name: str, collection_name: str):
    collection = get_or_create_chroma_collection(collection_name)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=400, detail=f"Lỗi khi mở ảnh: {e}")

    face = mtcnn(img, save_path=None)

    if face is None:
        raise HTTPException(status_code=400, detail=f"Không tìm thấy khuôn mặt nào trong ảnh cho {person_name}.")

    face = face.to(device)
    embedding = resnet(face).detach().cpu().numpy() # Chuyển tensor thành numpy array

    try:
        collection.upsert(
            embeddings=[embedding.flatten().tolist()], # Chuyển về list của float
            documents=[f"Embedding for {person_name}"],
            metadatas=[{"person_name": person_name}],
            ids=[person_name]
        )
        print(f"Đã thêm/cập nhật {person_name} vào cơ sở dữ liệu ChromaDB collection '{collection_name}'.")
        # Cập nhật lại bộ nhớ cache sau khi thêm
        load_database_to_memory(collection_name)
        return {"message": f"Đã thêm/cập nhật {person_name} vào collection '{collection_name}'."}
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm {person_name} vào ChromaDB: {e}")

# Hàm nhận diện khuôn mặt
# Chuyển hàm này thành đồng bộ để có thể chạy trong executor
def recognize_face_from_image_sync(image_bytes: bytes, collection_name: str, threshold: float = 0.65):
    # Tải lại database từ ChromaDB để đảm bảo cập nhật
    database = loaded_collections.get(collection_name)
    if database is None:
        database = load_database_to_memory(collection_name)
        if not database:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' không tồn tại hoặc không có dữ liệu.")

    try:
        img_attendance = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=400, detail=f"Lỗi khi mở ảnh điểm danh: {e}")

    faces = mtcnn(img_attendance, save_path=None)

    if faces is None:
        return {"recognized_faces": []}

    # Lấy bounding box, xác suất và landmarks
    bboxes, probs, landmarks = mtcnn.detect(img_attendance, landmarks=True)

    if bboxes is None or len(bboxes) == 0:
        return {"recognized_faces": []}

    if faces.dim() == 3:
        faces = faces.unsqueeze(0)

    # Đảm bảo số lượng khuôn mặt đã xử lý khớp với số lượng bounding box
    # Nếu không khớp, chỉ xử lý số lượng nhỏ hơn để tránh lỗi
    num_faces_to_process = min(len(bboxes), len(faces))
    
    embeddings = resnet(faces.to(device)).detach().cpu()

    recognized_results = []
    for i in range(num_faces_to_process):
        bbox = bboxes[i]
        current_embedding_cpu = embeddings[i]

        max_similarity = -1
        recognized_name = "Unknown"

        if not database:
            recognized_results.append({"name": recognized_name, "similarity": max_similarity, "bbox": bbox.tolist()})
            continue

        current_embedding_cpu = current_embedding_cpu.to(device)
        current_embedding_cpu = current_embedding_cpu / current_embedding_cpu.norm()

        for name, stored_embedding in database.items():
            if not isinstance(stored_embedding, torch.Tensor):
                print(f"Cảnh báo: Dữ liệu cho '{name}' trong DB không phải Tensor. Bỏ qua so sánh.")
                continue

            stored_embedding = stored_embedding / stored_embedding.norm()

            dot_product = torch.dot(current_embedding_cpu.flatten(), stored_embedding.flatten())
            norm_embedding = torch.norm(current_embedding_cpu)
            norm_stored = torch.norm(stored_embedding)

            if norm_embedding == 0 or norm_stored == 0:
                similarity_tensor = torch.tensor(0.0, device=device)
            else:
                similarity_tensor = dot_product / (norm_embedding * norm_stored)

            similarity = similarity_tensor.item()

            if similarity > max_similarity:
                max_similarity = similarity
                recognized_name = name

        if max_similarity >= threshold:
            recognized_results.append({"name": recognized_name, "similarity": round(max_similarity, 4), "bbox": bbox.tolist()})
        else:
            recognized_results.append({"name": "Unknown", "similarity": round(max_similarity, 4), "bbox": bbox.tolist()})

    return {"recognized_faces": recognized_results}

def get_faces_in_collection_sync(collection_name: str):
    """Lấy danh sách các khuôn mặt (IDs) có trong một collection ChromaDB."""
    collection = get_or_create_chroma_collection(collection_name)
    try:
        results = collection.get(
            ids=collection.get()['ids'],
            include=[] # Chỉ cần IDs
        )
        face_ids = results['ids']
        print(f"Đã lấy {len(face_ids)} khuôn mặt từ collection '{collection_name}'.")
        return {"face_ids": face_ids}
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách khuôn mặt từ ChromaDB collection '{collection_name}': {e}")

# --- Khởi tạo hàng đợi trong bộ nhớ ---
task_queue = asyncio.Queue()

# --- Hàm xử lý tác vụ nền ---
async def background_task_processor():
    while True:
        task_func, args, kwargs, future = await task_queue.get()
        try:
            # Chạy tác vụ trong một thread riêng để không chặn event loop
            result = await asyncio.to_thread(task_func, *args, **kwargs)
            future.set_result(result)
        except Exception as e:
            traceback.print_exc() # Print the full traceback
            future.set_exception(e)
        finally:
            task_queue.task_done()

# --- Khởi tạo FastAPI app ---
app = FastAPI(dependencies=[Depends(get_api_key)])


@app.on_event("startup")
async def startup_event():
    # Tải tất cả các collection hiện có vào bộ nhớ khi khởi động server
    # LƯU Ý: Với số lượng collection lớn, có thể cần chiến lược tải khác
    print("Đang tải tất cả các collection ChromaDB hiện có vào bộ nhớ...")
    all_collections = client.list_collections()
    for col_info in all_collections:
        load_database_to_memory(col_info.name)
    print("Đã tải xong các collection.")
    # Khởi động trình xử lý tác vụ nền
    asyncio.create_task(background_task_processor())


class EnrollFaceRequest(BaseModel):
    person_name: str
    collection_name: str

@app.post("/enroll_face/{collection_name}")
async def enroll_face_endpoint(collection_name: str, person_name: str = Form(...), file: UploadFile = File(...)):
    """
    Nạp khuôn mặt vào một collection cụ thể.
    - `collection_name`: Tên của collection ChromaDB để lưu trữ embedding.
    - `person_name`: Tên của người trong ảnh.
    - `file`: Ảnh khuôn mặt (dạng file upload).
    """
    image_bytes = await file.read()
    future = asyncio.Future()
    await task_queue.put((add_person_to_database_sync, (image_bytes, person_name, collection_name), {}, future))
    # Đợi tác vụ hoàn thành và trả về kết quả
    try:
        result = await future
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý yêu cầu nạp khuôn mặt: {e}")

@app.post("/recognize_face/{collection_name}")
async def recognize_face_endpoint(collection_name: str, file: UploadFile = File(...)):
    """
    Nhận diện khuôn mặt trên một collection cụ thể.
    - `collection_name`: Tên của collection ChromaDB để nhận diện.
    - `file`: Ảnh chứa khuôn mặt cần nhận diện (dạng file upload).
    """
    image_bytes = await file.read()
    future = asyncio.Future()
    await task_queue.put((recognize_face_from_image_sync, (image_bytes, collection_name), {}, future))
    # Đợi tác vụ hoàn thành và trả về kết quả
    try:
        result = await future
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý yêu cầu nhận diện khuôn mặt: {e}")

@app.get("/list_faces/{collection_name}")
async def list_faces_endpoint(collection_name: str):
    """
    Lấy danh sách các khuôn mặt (IDs) có trong một collection cụ thể.
    - `collection_name`: Tên của collection ChromaDB.
    """
    future = asyncio.Future()
    await task_queue.put((get_faces_in_collection_sync, (collection_name,), {}, future))
    try:
        result = await future
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc() # Print the full traceback
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý yêu cầu lấy danh sách khuôn mặt: {e}")