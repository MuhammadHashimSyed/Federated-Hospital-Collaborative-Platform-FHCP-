from rest_framework import viewsets, views, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import TrainingRound, LocalWeightSubmission, AuditLog
from .serializers import TrainingRoundSerializer, LocalWeightSubmissionSerializer, AuditLogSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

class CurrentRoundView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        current_round = TrainingRound.objects.filter(status__in=['PENDING', 'IN_PROGRESS', 'AGGREGATING']).order_by('-round_number').first()
        if not current_round:
            return Response({"detail": "No active round found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = TrainingRoundSerializer(current_round)
        data = serializer.data
        if request.user.role == 'HOSPITAL_CLIENT':
            data['has_submitted'] = LocalWeightSubmission.objects.filter(hospital=request.user, training_round=current_round).exists()
        return Response(data)

class UploadWeightsView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'HOSPITAL_CLIENT':
            return Response({"detail": "Only hospital clients can upload weights."}, status=status.HTTP_403_FORBIDDEN)
        
        current_round = TrainingRound.objects.filter(status='IN_PROGRESS').first()
        if not current_round:
            return Response({"detail": "No round is currently in progress."}, status=status.HTTP_400_BAD_REQUEST)
        
        submission = LocalWeightSubmission.objects.filter(hospital=request.user, training_round=current_round).first()
        if submission:
            serializer = LocalWeightSubmissionSerializer(submission, data=request.data)
        else:
            serializer = LocalWeightSubmissionSerializer(data=request.data)
            
        if serializer.is_valid():
            serializer.save(hospital=request.user, training_round=current_round)
            
            # Log the submission
            AuditLog.objects.create(
                action=f"Weights submitted for Round {current_round.round_number}",
                actor=request.user,
                ip_address=request.META.get('REMOTE_ADDR')
            )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AnalyticsMetricsView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        rounds = TrainingRound.objects.filter(status='COMPLETED').order_by('round_number')
        data = []
        for r in rounds:
            round_data = {
                "round_number": r.round_number,
                "global_mean_auroc": r.global_mean_auroc,
                "global_metrics": {},
                "local_submissions": []
            }
            submissions = LocalWeightSubmission.objects.filter(training_round=r)
            
            import json
            class_sums = {}
            class_counts = {}
            
            for sub in submissions:
                round_data["local_submissions"].append({
                    "hospital": sub.hospital.hospital_name or sub.hospital.username,
                    "local_mean_auroc": sub.local_mean_auroc,
                    "metrics_json": sub.metrics_json
                })
                if sub.metrics_json:
                    try:
                        metrics = json.loads(sub.metrics_json)
                        for k, v in metrics.items():
                            class_sums[k] = class_sums.get(k, 0) + v
                            class_counts[k] = class_counts.get(k, 0) + 1
                    except Exception:
                        pass
                        
            for k in class_sums:
                round_data["global_metrics"][k] = class_sums[k] / class_counts[k]
                
            data.append(round_data)
        return Response(data)

class StartRoundView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can start a round."}, status=status.HTTP_403_FORBIDDEN)

        # Block new round if any round is still active (not yet aggregated)
        active_round = TrainingRound.objects.filter(
            status__in=['IN_PROGRESS', 'AGGREGATING', 'PENDING']
        ).first()
        if active_round:
            return Response(
                {
                    "detail": f"Round {active_round.round_number} is still {active_round.get_status_display()}. "
                              f"Please run FedAvg aggregation to complete it before starting a new round."
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        latest_round = TrainingRound.objects.order_by('-round_number').first()
        next_number = 1 if not latest_round else latest_round.round_number + 1

        new_round = TrainingRound.objects.create(
            round_number=next_number,
            status='IN_PROGRESS'
        )
        AuditLog.objects.create(
            action=f"Started Round {next_number}",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )
        return Response(TrainingRoundSerializer(new_round).data, status=status.HTTP_201_CREATED)

class LiveDiagnosticView(views.APIView):
    """Real-time chest X-Ray diagnostic inference using the trained DenseNet121 federated model.
    Only returns pathologies with a sigmoid probability greater than 69%.
    """
    permission_classes = [IsAuthenticated]

    # CheXpert 14-class labels (matches the training label order)
    LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    # Minimum confidence threshold for a finding to appear in the report
    CONFIDENCE_THRESHOLD = 0.69

    _model = None
    _device = None

    @classmethod
    def _load_model(cls):
        """Loads and caches the DenseNet121 model from the fyp_model directory."""
        if cls._model is not None:
            return cls._model, cls._device

        import torch
        import torch.nn as nn
        from torchvision import models as tv_models
        import os

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls._device = device

        # Resolve model path relative to this file → up two levels → fyp_model/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, '..', 'fyp_model', 'backup_round_5.pth')
        model_path = os.path.normpath(model_path)

        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        model.to(device)
        model.eval()
        cls._model = model
        return cls._model, cls._device

    @staticmethod
    def _preprocess(image_file):
        """Applies the same preprocessing pipeline used during federated training."""
        import torch
        from torchvision import transforms
        from torchvision.transforms import functional as TF
        from PIL import Image

        class SquarePad:
            def __call__(self, img):
                w, h = img.size
                max_wh = max(w, h)
                p_left = (max_wh - w) // 2
                p_top = (max_wh - h) // 2
                p_right = max_wh - w - p_left
                p_bottom = max_wh - h - p_top
                return TF.pad(img, (p_left, p_top, p_right, p_bottom), 0, 'constant')

        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_file).convert('RGB')
        return transform(image).unsqueeze(0)

    def post(self, request):
        if 'image' not in request.FILES:
            return Response({"detail": "Please upload an image."}, status=status.HTTP_400_BAD_REQUEST)

        import torch

        try:
            model, device = self._load_model()
        except FileNotFoundError as e:
            return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            return Response({"detail": f"Model loading error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            img_tensor = self._preprocess(request.FILES['image']).to(device)
        except Exception as e:
            return Response({"detail": f"Image preprocessing error: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]

        raw_scores = dict(zip(self.LABELS, [float(p) for p in probs]))

        # --- Medical logic: suppress 'No Finding' if any real pathology fires above threshold ---
        is_sick = any(
            raw_scores[label] > self.CONFIDENCE_THRESHOLD
            for label in self.LABELS
            if label != 'No Finding'
        )
        if is_sick:
            raw_scores['No Finding'] = 0.0

        # --- Filter: only keep findings above 69% confidence ---
        high_confidence = {
            label: round(prob, 4)
            for label, prob in raw_scores.items()
            if prob > self.CONFIDENCE_THRESHOLD
        }

        # Sort descending by probability
        high_confidence = dict(
            sorted(high_confidence.items(), key=lambda x: x[1], reverse=True)
        )

        return Response({
            "predictions": high_confidence,
            "threshold": self.CONFIDENCE_THRESHOLD,
            "total_findings": len(high_confidence)
        }, status=status.HTTP_200_OK)


class ManualAggregateView(views.APIView):
    """Allows a Super Admin to manually trigger FedAvg aggregation for the current round."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can trigger aggregation."}, status=status.HTTP_403_FORBIDDEN)

        current_round = TrainingRound.objects.filter(status='IN_PROGRESS').first()
        if not current_round:
            return Response({"detail": "No round is currently IN_PROGRESS. Start a new round first."}, status=status.HTTP_400_BAD_REQUEST)

        submissions = LocalWeightSubmission.objects.filter(training_round=current_round)
        if not submissions.exists():
            return Response({"detail": "No hospital has submitted weights yet for this round."}, status=status.HTTP_400_BAD_REQUEST)

        # Import and run real FedAvg
        from .tasks import aggregate_weights_for_round
        result = aggregate_weights_for_round(current_round.id)

        AuditLog.objects.create(
            action=f"Admin manually triggered FedAvg aggregation for Round {current_round.round_number} ({submissions.count()} submissions)",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )

        # Re-fetch to get updated status
        current_round.refresh_from_db()
        return Response({
            "detail": f"FedAvg completed for Round {current_round.round_number}.",
            "global_mean_auroc": current_round.global_mean_auroc,
            "submissions_aggregated": submissions.count(),
            "status": current_round.status
        }, status=status.HTTP_200_OK)


class RegisteredHospitalsView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can view registered hospitals."}, status=status.HTTP_403_FORBIDDEN)
        
        from users.models import User
        hospitals = User.objects.filter(role='HOSPITAL_CLIENT').values('id', 'username', 'hospital_name', 'date_joined')
        return Response(list(hospitals))

class LocalHistoryView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'HOSPITAL_CLIENT':
            return Response({"detail": "Only hospital clients have local history."}, status=status.HTTP_403_FORBIDDEN)
            
        submissions = LocalWeightSubmission.objects.filter(hospital=request.user).order_by('-training_round__round_number')
        data = []
        for sub in submissions:
            data.append({
                "round_number": sub.training_round.round_number,
                "local_mean_auroc": sub.local_mean_auroc,
                "metrics_json": sub.metrics_json,
                "submitted_at": sub.submitted_at
            })
        return Response(data)

class ResetPlatformView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can reset the platform."}, status=status.HTTP_403_FORBIDDEN)
            
        TrainingRound.objects.all().delete()
        AuditLog.objects.create(
            action="Platform Reset (Started from Round 1)",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )
        return Response({"detail": "Platform reset successfully."}, status=status.HTTP_200_OK)

class SystemLogsView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can view system logs."}, status=status.HTTP_403_FORBIDDEN)
            
        logs = AuditLog.objects.all().order_by('-timestamp')
        data = []
        for log in logs:
            data.append({
                "timestamp": log.timestamp,
                "actor": log.actor.username if log.actor else "System",
                "action": log.action,
                "ip_address": log.ip_address
            })
        return Response(data)

class ClientLogView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        action = request.data.get('action', 'Unknown action')
        AuditLog.objects.create(
            action=action,
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )
        return Response({"detail": "Logged successfully"}, status=status.HTTP_201_CREATED)


class RegisterHospitalView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can register hospitals."}, status=status.HTTP_403_FORBIDDEN)

        username = request.data.get('username')
        password = request.data.get('password')
        hospital_name = request.data.get('hospital_name', '')

        if not username or not password:
            return Response({"detail": "Username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({"detail": f"Username '{username}' already exists."}, status=status.HTTP_400_BAD_REQUEST)

        hospital = User.objects.create_user(
            username=username,
            password=password,
            hospital_name=hospital_name,
            role='HOSPITAL_CLIENT'
        )

        AuditLog.objects.create(
            action=f"Registered new hospital node: {username} ({hospital_name})",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )

        return Response({"detail": f"Hospital '{hospital_name or username}' registered successfully.", "id": hospital.id}, status=status.HTTP_201_CREATED)


class DeleteHospitalView(views.APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, hospital_id):
        if request.user.role != 'SUPER_ADMIN':
            return Response({"detail": "Only super admins can delete hospitals."}, status=status.HTTP_403_FORBIDDEN)

        try:
            hospital = User.objects.get(id=hospital_id, role='HOSPITAL_CLIENT')
            name = hospital.username
            hospital.delete()
            AuditLog.objects.create(
                action=f"Deleted hospital node: {name}",
                actor=request.user,
                ip_address=request.META.get('REMOTE_ADDR')
            )
            return Response({"detail": f"Hospital '{name}' deleted successfully."}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"detail": "Hospital not found."}, status=status.HTTP_404_NOT_FOUND)


class DownloadGlobalModelView(views.APIView):
    """Serves the latest aggregated global model .pth file to authenticated hospital clients."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        import os
        from django.http import FileResponse
        from django.conf import settings

        latest_round = TrainingRound.objects.filter(
            status='COMPLETED',
            global_model_file__isnull=False
        ).order_by('-round_number').first()

        if not latest_round:
            return Response(
                {"detail": "No global model available yet. Complete at least one round of FedAvg first."},
                status=status.HTTP_404_NOT_FOUND
            )

        model_path = os.path.join(settings.MEDIA_ROOT, latest_round.global_model_file.name)

        if not os.path.exists(model_path):
            return Response(
                {"detail": "Global model file not found on server disk."},
                status=status.HTTP_404_NOT_FOUND
            )

        AuditLog.objects.create(
            action=f"Downloaded global model from Round {latest_round.round_number}",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )

        f = open(model_path, 'rb')
        response = FileResponse(f, content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="global_model_round_{latest_round.round_number}.pth"'
        response['X-Round-Number'] = str(latest_round.round_number)
        response['X-Global-AUROC'] = str(latest_round.global_mean_auroc)
        return response


class HospitalDoctorListView(views.APIView):
    """Hospital client can list and register doctors under their node."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'HOSPITAL_CLIENT':
            return Response({"detail": "Only hospital clients can manage doctors."}, status=status.HTTP_403_FORBIDDEN)
        doctors = User.objects.filter(role='DOCTOR', linked_hospital=request.user)
        data = [
            {"id": d.id, "username": d.username, "full_name": d.get_full_name() or d.username, "date_joined": d.date_joined}
            for d in doctors
        ]
        return Response(data)

    def post(self, request):
        if request.user.role != 'HOSPITAL_CLIENT':
            return Response({"detail": "Only hospital clients can register doctors."}, status=status.HTTP_403_FORBIDDEN)

        username = request.data.get('username', '').strip()
        password = request.data.get('password', '').strip()
        full_name = request.data.get('full_name', '').strip()

        if not username or not password:
            return Response({"detail": "Username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({"detail": f"Username '{username}' is already taken."}, status=status.HTTP_400_BAD_REQUEST)

        first_name, *last = full_name.split(' ', 1)
        last_name = last[0] if last else ''

        doctor = User.objects.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role='DOCTOR',
            linked_hospital=request.user
        )

        AuditLog.objects.create(
            action=f"Registered doctor: {username} ({full_name}) under {request.user.hospital_name or request.user.username}",
            actor=request.user,
            ip_address=request.META.get('REMOTE_ADDR')
        )

        return Response({"detail": f"Dr. {full_name or username} registered successfully.", "id": doctor.id}, status=status.HTTP_201_CREATED)


class HospitalDoctorDeleteView(views.APIView):
    """Hospital client can remove a doctor linked to their node."""
    permission_classes = [IsAuthenticated]

    def delete(self, request, doctor_id):
        if request.user.role != 'HOSPITAL_CLIENT':
            return Response({"detail": "Only hospital clients can remove doctors."}, status=status.HTTP_403_FORBIDDEN)

        try:
            doctor = User.objects.get(id=doctor_id, role='DOCTOR', linked_hospital=request.user)
            name = doctor.username
            doctor.delete()
            AuditLog.objects.create(
                action=f"Removed doctor: {name} from {request.user.hospital_name or request.user.username}",
                actor=request.user,
                ip_address=request.META.get('REMOTE_ADDR')
            )
            return Response({"detail": f"Dr. {name} removed successfully."}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"detail": "Doctor not found or not linked to your hospital."}, status=status.HTTP_404_NOT_FOUND)
