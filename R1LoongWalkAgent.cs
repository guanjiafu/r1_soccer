using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;

/// <summary>
/// 参考 LoongPlay 的稳定走路训练方式，为 R1 重新做一个简洁版 locomotion。
/// 重点：不额外罚关节动作/身体角速度，只靠：
///   + 前进速度
///   - 侧向速度
///   - yaw 跟踪
///   - 小 roll/yaw 角度惩罚
///   + hip roll 钳制
/// </summary>
public class R1LoongWalkAgent : Agent
{
    [Header("Mode")]
    public bool train = true;
    public bool fixbody = false;
    public bool autoChase = true;

    [Header("Reference")]
    public Transform ball;

    [Header("Command")]
    public float wr = 0f;

    [Header("Gait")]
    public int T1 = 30;
    public float dh = 30f;
    public float d0 = 5f;
    public float hipRollLimit = 10f;

    int tp = 0;
    int tt = 0;
    int nextDebugLogStep = 0;
    float uf1 = 0f;
    float uf2 = 0f;

    float[] u = new float[12];
    float[] utotal = new float[12];

    Transform body;
    int ActionNum;
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody[] acts = new ArticulationBody[20];

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    Vector3 pos0;
    Quaternion rot0;

    // R1 关节顺序的反馈系数，参考 Loong 的 kb 映射
    // 0 right_hip_pitch,1 right_hip_roll,2 right_hip_yaw,3 right_knee,4 right_ankle_pitch,5 right_ankle_roll
    float[] kb = new float[12]
    {
        30, 10, 20, 10, 30, 10,
        30, 10, 20, 10, 30, 10
    };

    public override void Initialize()
    {
        arts = GetComponentsInChildren<ArticulationBody>();
        ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if (arts[k].jointType == ArticulationJointType.RevoluteJoint)
            {
                acts[ActionNum] = arts[k];
                ActionNum++;
                if (ActionNum >= acts.Length) break;
            }
        }

        body = arts[0].GetComponent<Transform>();
        pos0 = body.position;
        rot0 = body.rotation;
        arts[0].GetJointPositions(P0);
        arts[0].GetJointVelocities(W0);
    }

    private bool _isClone = false;

    void Start()
    {
        Time.fixedDeltaTime = 0.01f;
        if (train && !_isClone)
        {
            for (int i = 1; i < 14; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.transform.position = transform.position + new Vector3(i * 2f, 0f, 0f);
                clone.name = $"{name}_Clone_{i}";
                clone.GetComponent<R1LoongWalkAgent>()._isClone = true;
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        tp = 0;
        tt = 0;
        for (int i = 0; i < 12; i++) u[i] = 0f;

        if (fixbody)
        {
            arts[0].immovable = true;
            arts[0].TeleportRoot(pos0 + Vector3.up * 0.2f, rot0);
        }
        else
        {
            arts[0].TeleportRoot(pos0, rot0);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            arts[0].SetJointPositions(P0);
            arts[0].SetJointVelocities(W0);
        }

        wr = 0f;
        if (train && Random.Range(0, 2) == 0)
        {
            wr = Random.Range(0.3f, 0.8f) * (Random.Range(0, 2) * 2 - 1);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(body.InverseTransformDirection(Vector3.down));
        sensor.AddObservation(body.InverseTransformDirection(arts[0].angularVelocity));
        sensor.AddObservation(body.InverseTransformDirection(arts[0].velocity));
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f);

        for (int i = 0; i < ActionNum; i++)
        {
            sensor.AddObservation(acts[i].jointPosition[0]);
            sensor.AddObservation(acts[i].jointVelocity[0]);
        }

        sensor.AddObservation(wr);
    }

    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180f) return eulerAngle;
        return eulerAngle - 360f;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        for (int i = 0; i < 12; i++) utotal[i] = 0f;
        var continuousActions = actionBuffers.ContinuousActions;
        float kk = 0.9f;

        for (int i = 0; i < ActionNum; i++)
        {
            u[i] = u[i] * kk + (1f - kk) * continuousActions[i];
            utotal[i] = kb[i] * u[i];
        }

        float g1 = dh * uf1 + d0;
        float g2 = dh * uf2 + d0;

        utotal[0] += -g1;   // right_hip_pitch
        utotal[3] += 2f * g1; // right_knee
        utotal[4] += -g1;   // right_ankle_pitch

        utotal[6] += -g2;   // left_hip_pitch
        utotal[9] += 2f * g2; // left_knee
        utotal[10] += -g2;  // left_ankle_pitch

        // 钳住 hip roll，防止腿越张越开
        utotal[1] = Mathf.Clamp(utotal[1], -hipRollLimit, hipRollLimit);
        utotal[7] = Mathf.Clamp(utotal[7], -hipRollLimit, hipRollLimit);

        for (int i = 0; i < ActionNum; i++) SetJointTargetDeg(acts[i], utotal[i]);
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 2000f;
        drive.damping = 100f;
        drive.forceLimit = 300f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        for (int i = 0; i < continuous.Length; i++) continuous[i] = 0f;
    }

    void FixedUpdate()
    {
        if (!train && autoChase && ball != null)
        {
            Vector3 toBall = ball.position - body.position;
            toBall.y = 0f;
            Vector3 forward = body.forward;
            forward.y = 0f;
            float angleDiff = Vector3.SignedAngle(forward.normalized, toBall.normalized, Vector3.up);
            wr = Mathf.Clamp(angleDiff * 0.3f, -0.8f, 0.8f);
        }

        tp++;
        if (tp > 0 && tp <= T1)
        {
            uf1 = (-Mathf.Cos(3.14f * 2f * tp / T1) + 1f) / 2f;
            uf2 = 0f;
        }
        if (tp > T1 && tp <= 2 * T1)
        {
            int tp0 = tp - T1;
            uf1 = 0f;
            uf2 = (-Mathf.Cos(3.14f * 2f * tp0 / T1) + 1f) / 2f;
        }
        if (tp >= 2 * T1) tp = 0;

        tt++;

        var vel = body.InverseTransformDirection(arts[0].velocity);
        var wel = body.InverseTransformDirection(arts[0].angularVelocity);

        float live = 1f;
        // euler[0]=pitch(前后倾), euler[2]=roll(左右倾)；参考 Loong 各 -0.1，不额外罚角速度
        float pitchPenalty = -0.1f * Mathf.Abs(EulerTrans(body.eulerAngles[0]));
        float rollPenalty = -0.1f * Mathf.Abs(EulerTrans(body.eulerAngles[2]));
        float welReward = -Mathf.Abs(wel[1] - wr);
        float velReward = vel[2] - Mathf.Abs(vel[0]);

        float ko = 0.4f;
        float kw = 1f;
        if (tt > 900)
        {
            ko = 1f;
            kw = 4f;
        }

        AddReward(live + (pitchPenalty + rollPenalty) * ko + welReward * kw + velReward);

        if (Mathf.Abs(EulerTrans(body.eulerAngles[0])) > 20f ||
            Mathf.Abs(EulerTrans(body.eulerAngles[2])) > 20f ||
            tt >= 1000)
        {
            if (train && !_isClone && Academy.Instance.StepCount >= nextDebugLogStep)
            {
                Debug.Log($"[R1LoongWalk END] step={Academy.Instance.StepCount} episodeLen={tt} pitch={EulerTrans(body.eulerAngles[0]):F2} roll={EulerTrans(body.eulerAngles[2]):F2}");
                nextDebugLogStep = Academy.Instance.StepCount + 20000;
            }

            if (train) EndEpisode();
        }
    }
}
