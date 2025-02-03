# load_test.py
import time
import json
from locust import HttpUser, task, between
from datetime import datetime
import os
import random

# Ensure test_images directory exists
TEST_IMAGES_DIR = "test_images"
if not os.path.exists(TEST_IMAGES_DIR):
    os.makedirs(TEST_IMAGES_DIR)


class BottleClassifierLoadTest(HttpUser):
    wait_time = between(0.001, 0.002)

    def on_start(self):
        """Initialize test data"""
        # List to store available test images
        self.test_images = []

        # Load test images
        for filename in os.listdir(TEST_IMAGES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self.test_images.append(os.path.join(TEST_IMAGES_DIR, filename))

        if not self.test_images:
            raise Exception("No test images found in directory!")

        # Initialize results log
        self.log_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    @task(1)
    def classify_image(self):
        """Test the classify endpoint"""
        try:
            # Select random test image
            image_path = random.choice(self.test_images)

            start_time = time.time()

            # Send request to correct endpoint
            with open(image_path, 'rb') as image_file:
                response = self.client.post(
                    "/classify",  # Correct endpoint
                    files={'image': image_file},
                    name="/classify"
                )

            request_time = time.time() - start_time

            # Log results
            if response.status_code == 200:
                self.log_result(
                    request_time=request_time,
                    success=True,
                    response_data=response.json()
                )
            else:
                self.log_result(
                    request_time=request_time,
                    success=False,
                    error=f"Status code: {response.status_code}",
                    response_data=response.text
                )

        except Exception as e:
            self.log_result(
                request_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

class ResultAnalyzer:
    """Analyze load test results"""

    def __init__(self, log_file):
        self.log_file = log_file

    def analyze(self):
        """Analyze test results"""
        total_requests = 0
        successful_requests = 0
        total_time = 0
        response_times = []

        with open(self.log_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                total_requests += 1
                if data['success']:
                    successful_requests += 1
                total_time += data['request_time']
                response_times.append(data['request_time'])

        # Calculate metrics
        success_rate = (successful_requests / total_requests) * 100
        avg_response_time = total_time / total_requests
        response_times.sort()
        p95_response_time = response_times[int(len(response_times) * 0.95)]

        return {
            'total_requests': total_requests,
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'requests_per_second': total_requests / total_time
        }


if __name__ == "__main__":
    print("To run the load test:")
    print("1. Place test images in './test_images' directory")
    print("2. Run: locust -f load_test.py --host=http://your-api-url")
    print("3. Open http://localhost:8089 in your browser")
    print("4. Set number of users and spawn rate")
    print("5. Start the test")
    print("\nAfter testing, analyze results with:")
    print("analyzer = ResultAnalyzer('your_log_file.json')")
    print("results = analyzer.analyze()")