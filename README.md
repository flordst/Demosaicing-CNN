# Demosaicing with Deep Learning


HƯỚNG DẪN CHẠY CODE 
1.	Thực hiện huấn luyện và kiểm thử model
Sau khi tải source code và bộ dữ liệu về thì thực hiện theo các bước sau 
B1 :Upload source code lên google drive
B2: Đưa thư mục CUB_200_2011 vào thư mục “data” 
B3 : Khởi động google colab, liên kết với source code ,sau đó chạy lần lượt các lệnh như sau :
-Tiền xử lý dữ liệu:
!python preprocess.py  --crop false
-Huấn luyện mô hình:
!python train_main.py --resume_from_ckp <true or false>  --trial_number <an integer> --model <model name>  --num_epochs <an integer>  --lr <a float number> --not_cropped true
Trong đó :
<model name> :SRCNN or rednet20
num_epochs :số vòng huấn luyện
lr : Tốc độ học
-Kiểm thử :
!python testing_main.py --trial_number <an integer> --model <model name>
Kết quả sẽ được lưu trong thư mục ‘data_outs’
	Lưu ý : Khi muốn kiểm thử với mô hình khác ,hãy di chuyển kết quả của mô hình hiện tại sang một nơi khác .

2.	Đánh giá kết quả và so sánh các phương pháp 
-Chạy code  “Danh_gia_CNN.ipynb”  để tính PSNR và SSIM của 2 phương pháp CNN

Để đánh giá kết quả của các phương pháp truyền thống ta làm như sau :
-Tải kết quả sau khi kiểm thử của 2 mô hình về , sau đó lọc ra tập dữ liệu groundtruth 
-Chạy các file sau trên matlab :


