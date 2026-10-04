import io
import numpy as np
from minio import Minio
from minio.error import S3Error

class SignPoseStorage:
    def __init__(self, endpoint="localhost:9000", access_key="admin", secret_key="password123", secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self._ensure_core_buckets()

    def _ensure_core_buckets(self):
        """Validates that the essential buckets exist, creating them if necessary."""
        #required_buckets = ["raw-data", "vocabulary-videos", "vocabulary-poses", "stitched-sequences"]
        required_buckets = ["raw-data"]
        for bucket in required_buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def put_pose_sequence(self, bucket_name: str, object_name: str, pose_tensor: np.ndarray):
        """
        Uploads a 3D coordinate sequence directly from memory to MinIO.
        Expected tensor shape: (frames, 152)
        """
        try:
            buffer = io.BytesIO()
            np.save(buffer, pose_tensor)
            buffer.seek(0)
            
            self.client.put_object(
                bucket_name,
                f"{object_name}.npy",
                data=buffer,
                length=buffer.getbuffer().nbytes
            )
        except S3Error as err:
            print(f"MinIO Upload Error on {object_name}: {err}")

    def get_pose_sequence(self, bucket_name: str, object_name: str) -> np.ndarray:
        """
        Retrieves a pose sequence from MinIO and loads it directly into a NumPy array.
        """
        response = None
        try:
            if not object_name.endswith('.npy'):
                object_name += '.npy'
                
            response = self.client.get_object(bucket_name, object_name)
            buffer = io.BytesIO(response.read())
            return np.load(buffer)
        except S3Error as err:
            print(f"MinIO Retrieval Error on {object_name}: {err}")
            return np.array([])
        finally:
            if response:
                response.close()
                response.release_conn()

    def list_objects(self, bucket_name: str, suffix: str = "") -> list:
        """Returns a list of object names within a specific bucket, filtered by suffix."""
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            return [obj.object_name for obj in objects if obj.object_name.endswith(suffix)]
        except S3Error as err:
            print(f"MinIO List Error: {err}")
            return []

    def download_file(self, bucket_name: str, object_name: str, local_path: str):
        """Downloads a file directly to the local file system."""
        try:
            self.client.fget_object(bucket_name, object_name, local_path)
        except S3Error as err:
            print(f"MinIO Download Error on {object_name}: {err}")

    def upload_file(self, bucket_name: str, object_name: str, local_path: str):
        """Uploads a file directly from the local file system."""
        try:
            self.client.fput_object(bucket_name, object_name, local_path)
        except S3Error as err:
            print(f"MinIO File Upload Error on {object_name}: {err}")