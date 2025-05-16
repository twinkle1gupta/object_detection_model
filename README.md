### 📝 Experience Report

*Challenges:*
I ran into shape mismatch issues when connecting the VGG16 backbone to the YOLO detection head. Also, due to macOS, I faced some version compatibility problems with PyTorch and OpenCV, which took time to resolve.

*AI Tools Used:*
I used ChatGPT to understand YOLO's architecture and debug tensor shapes. GitHub Copilot also helped fill in some boilerplate code and suggested function templates. Both were helpful but didn’t replace hands-on debugging.

*What I Learned:*
I got a much deeper understanding of how YOLO handles bounding boxes, objectness, and class probabilities. Also learned the importance of matching grid sizes and tensor dimensions in object detection.

*What Surprised Me:*
How fragile everything is — a small mistake in reshaping tensors can break the whole model. Also, how much detail goes into encoding labels.

*Segmentation/Data Prep:*
Surprisingly, the annotation segmentation and CSV-based dataset loading went smoother than I expected. The grid mapping logic was tricky at first but became manageable.

*Code vs. AI Help Balance:*
AI tools gave me momentum, especially early on. But understanding and debugging still required my full attention. The balance felt fair — assistive, not replacement.

*Suggestions:*
Optional starter code for the evaluation metrics (to save time reinventing the wheel)
An example of one training/validation step with expected tensor shapes, so we can debug more easily
