/*
 * CavLyriqModel_types.h
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "CavLyriqModel".
 *
 * Model version              : 6.37
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Thu Jul 25 03:15:01 2024
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Execution efficiency
 * Validation result: Not run
 */

#ifndef RTW_HEADER_CavLyriqModel_types_h_
#define RTW_HEADER_CavLyriqModel_types_h_
#include "rtwtypes.h"
#ifndef DEFINED_TYPEDEF_FOR_DriverCommandsBus_
#define DEFINED_TYPEDEF_FOR_DriverCommandsBus_

struct DriverCommandsBus
{
  real_T SteerCmd;
  real_T AccelCmd;
  real_T DecelCmd;
  real_T GearCmd;
  boolean_T IgnSwtch;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_CavCommandsBus_
#define DEFINED_TYPEDEF_FOR_CavCommandsBus_

struct CavCommandsBus
{
  real_T CavAxlTrqReq_Nm;
  real_T CavBrkReq_mps2;
  real_T CavSteerTrq_Nm;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_EngineBus_
#define DEFINED_TYPEDEF_FOR_EngineBus_

struct EngineBus
{
  real_T EngTrq;
  real_T EngSpdOut;
  real_T Map;
  real_T Mat;
  real_T ThrPosPct;
  real_T WgAreaPct;
  real_T IntkCamPhase;
  real_T ExhCamPhase;
  real_T TurboRackPos;
  real_T EgrVlvAreaPct;
  real_T VarCompRatioPos;
  real_T IntkVlvLift;
  real_T VarIntkRunLen;
  real_T IntkSwirlVlvPos;
  real_T FuelFlw;
  real_T FuelVolFlw;
  real_T IntkPortFlw;
  real_T NormAirChrg;
  real_T ExhManGasTemp;
  real_T Afr;
  real_T TurboSpd;
  real_T TurbPrsRatio;
  real_T CompPrsRatio;
  real_T TurbTempOut;
  real_T CompTempOut;
  real_T EgrPct;
  real_T EgrMassFlwRate;
  real_T EgrCoolerTempOut;
  real_T IntercoolerTempOut;
  real_T BSFC;
  real_T TPHC;
  real_T TPCO;
  real_T TPNOx;
  real_T TPCO2;
  real_T Iat;
  real_T Ect;
  real_T EgrVlvDeltaPrs;
  real_T EgrVlvInTemp;
  real_T CylPrs;
  real_T EngTrqCrk;
  real_T CrkAng;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_TransBus_
#define DEFINED_TYPEDEF_FOR_TransBus_

struct TransBus
{
  real_T TransGear;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_TransInfoBus_
#define DEFINED_TYPEDEF_FOR_TransInfoBus_

struct TransInfoBus
{
  TransBus Trans;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus1_TransInfo_
#define DEFINED_TYPEDEF_FOR_slBus1_TransInfo_

struct slBus1_TransInfo
{
  real_T Cltch1State;
  real_T Cltch2State;
  TransInfoBus TransInfo;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_DrivelineBus_
#define DEFINED_TYPEDEF_FOR_DrivelineBus_

struct DrivelineBus
{
  real_T AxlTrq[4];
  real_T EngSpd;
  real_T DriveshftSpd[2];
  real_T EMSpd[2];
  slBus1_TransInfo TransInfo;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_BMSSensorsBus_
#define DEFINED_TYPEDEF_FOR_BMSSensorsBus_

struct BMSSensorsBus
{
  real32_T CellVoltages[96];
  real32_T PackVoltage;
  real32_T PackCurrent;
  real32_T CellTemperatures[96];
  real32_T VoutChgr;
  real32_T VoutInvtr;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus2_Battery_
#define DEFINED_TYPEDEF_FOR_slBus2_Battery_

struct slBus2_Battery
{
  real_T BattSoc;
  real_T BattCurr;
  real_T BattPwr;
  real_T BattVolt;
  BMSSensorsBus BMSSensors;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_EMBus_
#define DEFINED_TYPEDEF_FOR_EMBus_

struct EMBus
{
  real_T EMTrq[2];
  real_T EMPwrElec[2];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_FuelCellBus_
#define DEFINED_TYPEDEF_FOR_FuelCellBus_

struct FuelCellBus
{
  real_T FuelCellVolt;
  real_T FuelCellCurr;
  real_T H2Flw;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_DriverBus_
#define DEFINED_TYPEDEF_FOR_DriverBus_

struct DriverBus
{
  real_T SteerFdbk;
  real_T AccelFdbk;
  real_T DecelFdbk;
  real_T GearFdbk;
  boolean_T IgnFdbk;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_DispBus_
#define DEFINED_TYPEDEF_FOR_DispBus_

struct DispBus
{
  real_T X;
  real_T Y;
  real_T Z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_VelBus_
#define DEFINED_TYPEDEF_FOR_VelBus_

struct VelBus
{
  real_T Xdot;
  real_T Ydot;
  real_T Zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_AngBus_
#define DEFINED_TYPEDEF_FOR_AngBus_

struct AngBus
{
  real_T phi;
  real_T theta;
  real_T psi;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_CgBus_
#define DEFINED_TYPEDEF_FOR_CgBus_

struct CgBus
{
  DispBus Disp;
  VelBus Vel;
  AngBus Ang;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_LftBus_
#define DEFINED_TYPEDEF_FOR_LftBus_

struct LftBus
{
  DispBus Disp;
  VelBus Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_RghtBus_
#define DEFINED_TYPEDEF_FOR_RghtBus_

struct RghtBus
{
  DispBus Disp;
  VelBus Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_FrntAxlBus_
#define DEFINED_TYPEDEF_FOR_FrntAxlBus_

struct FrntAxlBus
{
  LftBus Lft;
  RghtBus Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_RearAxlBus_
#define DEFINED_TYPEDEF_FOR_RearAxlBus_

struct RearAxlBus
{
  LftBus Lft;
  RghtBus Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_HitchBus_
#define DEFINED_TYPEDEF_FOR_HitchBus_

struct HitchBus
{
  DispBus Disp;
  VelBus Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_GeomBus_
#define DEFINED_TYPEDEF_FOR_GeomBus_

struct GeomBus
{
  DispBus Disp;
  VelBus Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_InertFrmBus_
#define DEFINED_TYPEDEF_FOR_InertFrmBus_

struct InertFrmBus
{
  CgBus Cg;
  FrntAxlBus FrntAxl;
  RearAxlBus RearAxl;
  HitchBus Hitch;
  GeomBus Geom;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus3_Vel_
#define DEFINED_TYPEDEF_FOR_slBus3_Vel_

struct slBus3_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus4_Ang_
#define DEFINED_TYPEDEF_FOR_slBus4_Ang_

struct slBus4_Ang
{
  real_T Beta;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_AngVelBus_
#define DEFINED_TYPEDEF_FOR_AngVelBus_

struct AngVelBus
{
  real_T p;
  real_T q;
  real_T r;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_AccBus_
#define DEFINED_TYPEDEF_FOR_AccBus_

struct AccBus
{
  real_T ax;
  real_T ay;
  real_T az;
  real_T xddot;
  real_T yddot;
  real_T zddot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_AngAccBus_
#define DEFINED_TYPEDEF_FOR_AngAccBus_

struct AngAccBus
{
  real_T pdot;
  real_T qdot;
  real_T rdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus5_Cg_
#define DEFINED_TYPEDEF_FOR_slBus5_Cg_

struct slBus5_Cg
{
  slBus3_Vel Vel;
  slBus4_Ang Ang;
  AngVelBus AngVel;
  AccBus Acc;
  AngAccBus AngAcc;
  real_T DCM[9];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_BodyBus_
#define DEFINED_TYPEDEF_FOR_BodyBus_

struct BodyBus
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_ExtBus_
#define DEFINED_TYPEDEF_FOR_ExtBus_

struct ExtBus
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus6_Lft_
#define DEFINED_TYPEDEF_FOR_slBus6_Lft_

struct slBus6_Lft
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus7_Rght_
#define DEFINED_TYPEDEF_FOR_slBus7_Rght_

struct slBus7_Rght
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus8_FrntAxl_
#define DEFINED_TYPEDEF_FOR_slBus8_FrntAxl_

struct slBus8_FrntAxl
{
  slBus6_Lft Lft;
  slBus7_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus9_Lft_
#define DEFINED_TYPEDEF_FOR_slBus9_Lft_

struct slBus9_Lft
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus10_Rght_
#define DEFINED_TYPEDEF_FOR_slBus10_Rght_

struct slBus10_Rght
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus11_RearAxl_
#define DEFINED_TYPEDEF_FOR_slBus11_RearAxl_

struct slBus11_RearAxl
{
  slBus9_Lft Lft;
  slBus10_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus12_Hitch_
#define DEFINED_TYPEDEF_FOR_slBus12_Hitch_

struct slBus12_Hitch
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus13_Lft_
#define DEFINED_TYPEDEF_FOR_slBus13_Lft_

struct slBus13_Lft
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus14_Rght_
#define DEFINED_TYPEDEF_FOR_slBus14_Rght_

struct slBus14_Rght
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_FrntTiresBus_
#define DEFINED_TYPEDEF_FOR_FrntTiresBus_

struct FrntTiresBus
{
  slBus13_Lft Lft;
  slBus14_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus15_Lft_
#define DEFINED_TYPEDEF_FOR_slBus15_Lft_

struct slBus15_Lft
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus16_Rght_
#define DEFINED_TYPEDEF_FOR_slBus16_Rght_

struct slBus16_Rght
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_RearTiresBus_
#define DEFINED_TYPEDEF_FOR_RearTiresBus_

struct RearTiresBus
{
  slBus15_Lft Lft;
  slBus16_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_TiresBus_
#define DEFINED_TYPEDEF_FOR_TiresBus_

struct TiresBus
{
  FrntTiresBus FrntTires;
  RearTiresBus RearTires;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_DragBus_
#define DEFINED_TYPEDEF_FOR_DragBus_

struct DragBus
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_GrvtyBus_
#define DEFINED_TYPEDEF_FOR_GrvtyBus_

struct GrvtyBus
{
  real_T Fx;
  real_T Fy;
  real_T Fz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_ForcesBus_
#define DEFINED_TYPEDEF_FOR_ForcesBus_

struct ForcesBus
{
  BodyBus Body;
  ExtBus Ext;
  slBus8_FrntAxl FrntAxl;
  slBus11_RearAxl RearAxl;
  slBus12_Hitch Hitch;
  TiresBus Tires;
  DragBus Drag;
  GrvtyBus Grvty;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus17_Body_
#define DEFINED_TYPEDEF_FOR_slBus17_Body_

struct slBus17_Body
{
  real_T Mx;
  real_T My;
  real_T Mz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus18_Drag_
#define DEFINED_TYPEDEF_FOR_slBus18_Drag_

struct slBus18_Drag
{
  real_T Mx;
  real_T My;
  real_T Mz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus19_Ext_
#define DEFINED_TYPEDEF_FOR_slBus19_Ext_

struct slBus19_Ext
{
  real_T Mx;
  real_T My;
  real_T Mz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus20_Hitch_
#define DEFINED_TYPEDEF_FOR_slBus20_Hitch_

struct slBus20_Hitch
{
  real_T Mx;
  real_T My;
  real_T Mz;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_MomentsBus_
#define DEFINED_TYPEDEF_FOR_MomentsBus_

struct MomentsBus
{
  slBus17_Body Body;
  slBus18_Drag Drag;
  slBus19_Ext Ext;
  slBus20_Hitch Hitch;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus21_Disp_
#define DEFINED_TYPEDEF_FOR_slBus21_Disp_

struct slBus21_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus22_Vel_
#define DEFINED_TYPEDEF_FOR_slBus22_Vel_

struct slBus22_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus23_Lft_
#define DEFINED_TYPEDEF_FOR_slBus23_Lft_

struct slBus23_Lft
{
  slBus21_Disp Disp;
  slBus22_Vel Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus24_Disp_
#define DEFINED_TYPEDEF_FOR_slBus24_Disp_

struct slBus24_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus25_Vel_
#define DEFINED_TYPEDEF_FOR_slBus25_Vel_

struct slBus25_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus26_Rght_
#define DEFINED_TYPEDEF_FOR_slBus26_Rght_

struct slBus26_Rght
{
  slBus24_Disp Disp;
  slBus25_Vel Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus27_FrntAxl_
#define DEFINED_TYPEDEF_FOR_slBus27_FrntAxl_

struct slBus27_FrntAxl
{
  slBus23_Lft Lft;
  slBus26_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus28_Disp_
#define DEFINED_TYPEDEF_FOR_slBus28_Disp_

struct slBus28_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus29_Vel_
#define DEFINED_TYPEDEF_FOR_slBus29_Vel_

struct slBus29_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus30_Lft_
#define DEFINED_TYPEDEF_FOR_slBus30_Lft_

struct slBus30_Lft
{
  slBus28_Disp Disp;
  slBus29_Vel Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus31_Disp_
#define DEFINED_TYPEDEF_FOR_slBus31_Disp_

struct slBus31_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus32_Vel_
#define DEFINED_TYPEDEF_FOR_slBus32_Vel_

struct slBus32_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus33_Rght_
#define DEFINED_TYPEDEF_FOR_slBus33_Rght_

struct slBus33_Rght
{
  slBus31_Disp Disp;
  slBus32_Vel Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus34_RearAxl_
#define DEFINED_TYPEDEF_FOR_slBus34_RearAxl_

struct slBus34_RearAxl
{
  slBus30_Lft Lft;
  slBus33_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus35_Disp_
#define DEFINED_TYPEDEF_FOR_slBus35_Disp_

struct slBus35_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus36_Vel_
#define DEFINED_TYPEDEF_FOR_slBus36_Vel_

struct slBus36_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus37_Hitch_
#define DEFINED_TYPEDEF_FOR_slBus37_Hitch_

struct slBus37_Hitch
{
  slBus35_Disp Disp;
  slBus36_Vel Vel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_PwrBus_
#define DEFINED_TYPEDEF_FOR_PwrBus_

struct PwrBus
{
  real_T PwrExt;
  real_T Drag;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus38_Disp_
#define DEFINED_TYPEDEF_FOR_slBus38_Disp_

struct slBus38_Disp
{
  real_T x;
  real_T y;
  real_T z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus39_Vel_
#define DEFINED_TYPEDEF_FOR_slBus39_Vel_

struct slBus39_Vel
{
  real_T xdot;
  real_T ydot;
  real_T zdot;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus40_Ang_
#define DEFINED_TYPEDEF_FOR_slBus40_Ang_

struct slBus40_Ang
{
  real_T Beta;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus41_Geom_
#define DEFINED_TYPEDEF_FOR_slBus41_Geom_

struct slBus41_Geom
{
  slBus38_Disp Disp;
  slBus39_Vel Vel;
  slBus40_Ang Ang;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_BdyFrmBus_
#define DEFINED_TYPEDEF_FOR_BdyFrmBus_

struct BdyFrmBus
{
  slBus5_Cg Cg;
  ForcesBus Forces;
  MomentsBus Moments;
  slBus27_FrntAxl FrntAxl;
  slBus34_RearAxl RearAxl;
  slBus37_Hitch Hitch;
  PwrBus Pwr;
  slBus41_Geom Geom;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus42_Body_
#define DEFINED_TYPEDEF_FOR_slBus42_Body_

struct slBus42_Body
{
  InertFrmBus InertFrm;
  BdyFrmBus BdyFrm;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_WheelsBus_
#define DEFINED_TYPEDEF_FOR_WheelsBus_

struct WheelsBus
{
  real_T AxlTrq[4];
  real_T Omega[4];
  real_T Fx[4];
  real_T Fy[4];
  real_T Fz[4];
  real_T Mx[4];
  real_T My[4];
  real_T Mz[4];
  real_T Vx[4];
  real_T Vy[4];
  real_T Re[4];
  real_T Kappa[4];
  real_T Alpha[4];
  real_T a[4];
  real_T b[4];
  real_T Gamma[4];
  real_T psidot[4];
  real_T BrkTrq[4];
  real_T BrkPrs[4];
  real_T z[4];
  real_T zdot[4];
  real_T Gnd[4];
  real_T GndFz[4];
  real_T Prs[4];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_GndBus_
#define DEFINED_TYPEDEF_FOR_GndBus_

struct GndBus
{
  real_T mu[4];
  real_T z[4];
  real_T DCM[36];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus43_Wheels_
#define DEFINED_TYPEDEF_FOR_slBus43_Wheels_

struct slBus43_Wheels
{
  WheelsBus Wheels;
  GndBus Gnd;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_WhlAngBus_
#define DEFINED_TYPEDEF_FOR_WhlAngBus_

struct WhlAngBus
{
  real_T FL;
  real_T FR;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_FrontBus_
#define DEFINED_TYPEDEF_FOR_FrontBus_

struct FrontBus
{
  WhlAngBus WhlAng;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus44_WhlAng_
#define DEFINED_TYPEDEF_FOR_slBus44_WhlAng_

struct slBus44_WhlAng
{
  real_T RL;
  real_T RR;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_RearBus_
#define DEFINED_TYPEDEF_FOR_RearBus_

struct RearBus
{
  slBus44_WhlAng WhlAng;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SteeringBus_
#define DEFINED_TYPEDEF_FOR_SteeringBus_

struct SteeringBus
{
  FrontBus Front;
  RearBus Rear;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_VehBus_
#define DEFINED_TYPEDEF_FOR_VehBus_

struct VehBus
{
  real_T F[12];
  real_T M[12];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_WhlBus_
#define DEFINED_TYPEDEF_FOR_WhlBus_

struct WhlBus
{
  real_T F[12];
  real_T xdot[4];
  real_T ydot[4];
  real_T Ang[12];
  real_T z[4];
  real_T AngVel[12];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SuspChassisBus_
#define DEFINED_TYPEDEF_FOR_SuspChassisBus_

struct SuspChassisBus
{
  VehBus Veh;
  WhlBus Whl;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SuspWheelBus_
#define DEFINED_TYPEDEF_FOR_SuspWheelBus_

struct SuspWheelBus
{
  real_T WhlAng[12];
  real_T YawRate[4];
  real_T Vx[4];
  real_T Vy[4];
  real_T Vz[4];
  real_T Fext[4];
  real_T dz[4];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_SuspensionBus_
#define DEFINED_TYPEDEF_FOR_SuspensionBus_

struct SuspensionBus
{
  SuspChassisBus SuspChassis;
  SuspWheelBus SuspWheel;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus45_Disp_
#define DEFINED_TYPEDEF_FOR_slBus45_Disp_

struct slBus45_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus46_Lft_
#define DEFINED_TYPEDEF_FOR_slBus46_Lft_

struct slBus46_Lft
{
  slBus45_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus47_Disp_
#define DEFINED_TYPEDEF_FOR_slBus47_Disp_

struct slBus47_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus48_Rght_
#define DEFINED_TYPEDEF_FOR_slBus48_Rght_

struct slBus48_Rght
{
  slBus47_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus49_FrntAxl_
#define DEFINED_TYPEDEF_FOR_slBus49_FrntAxl_

struct slBus49_FrntAxl
{
  slBus46_Lft Lft;
  slBus48_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus50_Disp_
#define DEFINED_TYPEDEF_FOR_slBus50_Disp_

struct slBus50_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus51_Lft_
#define DEFINED_TYPEDEF_FOR_slBus51_Lft_

struct slBus51_Lft
{
  slBus50_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus52_Disp_
#define DEFINED_TYPEDEF_FOR_slBus52_Disp_

struct slBus52_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus53_Rght_
#define DEFINED_TYPEDEF_FOR_slBus53_Rght_

struct slBus53_Rght
{
  slBus52_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_MidlAxlBus_
#define DEFINED_TYPEDEF_FOR_MidlAxlBus_

struct MidlAxlBus
{
  slBus51_Lft Lft;
  slBus53_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus54_Disp_
#define DEFINED_TYPEDEF_FOR_slBus54_Disp_

struct slBus54_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus55_Lft_
#define DEFINED_TYPEDEF_FOR_slBus55_Lft_

struct slBus55_Lft
{
  slBus54_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus56_Disp_
#define DEFINED_TYPEDEF_FOR_slBus56_Disp_

struct slBus56_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus57_Rght_
#define DEFINED_TYPEDEF_FOR_slBus57_Rght_

struct slBus57_Rght
{
  slBus56_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus58_RearAxl_
#define DEFINED_TYPEDEF_FOR_slBus58_RearAxl_

struct slBus58_RearAxl
{
  slBus55_Lft Lft;
  slBus57_Rght Rght;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus59_Disp_
#define DEFINED_TYPEDEF_FOR_slBus59_Disp_

struct slBus59_Disp
{
  real_T X;
  real_T Y;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus60_Cg_
#define DEFINED_TYPEDEF_FOR_slBus60_Cg_

struct slBus60_Cg
{
  slBus59_Disp Disp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus61_InertFrm_
#define DEFINED_TYPEDEF_FOR_slBus61_InertFrm_

struct slBus61_InertFrm
{
  slBus49_FrntAxl FrntAxl;
  MidlAxlBus MidlAxl;
  slBus58_RearAxl RearAxl;
  slBus60_Cg Cg;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_TrlrBodyBus_
#define DEFINED_TYPEDEF_FOR_TrlrBodyBus_

struct TrlrBodyBus
{
  slBus61_InertFrm InertFrm;
  real_T Hitch[6];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_TrlrWheelsBus_
#define DEFINED_TYPEDEF_FOR_TrlrWheelsBus_

struct TrlrWheelsBus
{
  real_T Wheels;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus62_Engine_
#define DEFINED_TYPEDEF_FOR_slBus62_Engine_

struct slBus62_Engine
{
  real_T EngTrq;
  real_T Map;
  real_T Mat;
  real_T ThrPosPct;
  real_T WgAreaPct;
  real_T IntkCamPhase;
  real_T ExhCamPhase;
  real_T TurboRackPos;
  real_T EgrVlvAreaPct;
  real_T VarCompRatioPos;
  real_T IntkVlvLift;
  real_T VarIntkRunLen;
  real_T Afr;
  real_T IntkSwirlVlvPos;
  real_T TurboSpd;
  real_T Iat;
  real_T Ect;
  real_T EgrVlvDeltaPrs;
  real_T EgrVlvInTemp;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus63_Trans_
#define DEFINED_TYPEDEF_FOR_slBus63_Trans_

struct slBus63_Trans
{
  real_T Cltch1State;
  real_T Cltch2State;
  real_T DriveshftSpd[2];
  real_T AxlTrq[4];
  real_T EngSpd;
  real_T EMSpd[2];
  real_T TransGear;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus64_Battery_
#define DEFINED_TYPEDEF_FOR_slBus64_Battery_

struct slBus64_Battery
{
  real_T BattSoc;
  real_T BattVolt;
  real_T BattCurr;
  real_T BattPwr;
  BMSSensorsBus BMSSensors;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus65_Wheels_
#define DEFINED_TYPEDEF_FOR_slBus65_Wheels_

struct slBus65_Wheels
{
  real_T BrkPrs[4];
  real_T Kappa[4];
  real_T Alpha[4];
  real_T Omega[4];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_IMUBus_
#define DEFINED_TYPEDEF_FOR_IMUBus_

struct IMUBus
{
  real_T Acc[3];
  real_T pqr[3];
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_zBus_
#define DEFINED_TYPEDEF_FOR_zBus_

struct zBus
{
  real_T G_FL_z;
  real_T G_FR_z;
  real_T G_RL_z;
  real_T G_RR_z;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_muBus_
#define DEFINED_TYPEDEF_FOR_muBus_

struct muBus
{
  real_T mu_FL;
  real_T mu_FR;
  real_T mu_RL;
  real_T mu_RR;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_slBus66_Gnd_
#define DEFINED_TYPEDEF_FOR_slBus66_Gnd_

struct slBus66_Gnd
{
  zBus z;
  real_T DCM[36];
  muBus mu;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_EnvBus_
#define DEFINED_TYPEDEF_FOR_EnvBus_

struct EnvBus
{
  real_T Temp;
  real_T Baro;
  real_T Wind[3];
  real_T Grade;
  slBus66_Gnd Gnd;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_CntrlFdbkBus_
#define DEFINED_TYPEDEF_FOR_CntrlFdbkBus_

struct CntrlFdbkBus
{
  real_T Vehicle;
  DriverBus Driver;
  slBus62_Engine Engine;
  slBus63_Trans Trans;
  slBus64_Battery Battery;
  EMBus EM;
  FuelCellBus FuelCell;
  slBus65_Wheels Wheels;
  IMUBus IMU;
  EnvBus Env;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_VehFdbkBus_
#define DEFINED_TYPEDEF_FOR_VehFdbkBus_

struct VehFdbkBus
{
  EngineBus Engine;
  DrivelineBus Driveline;
  slBus2_Battery Battery;
  EMBus EM;
  FuelCellBus FuelCell;
  DriverBus Driver;
  slBus42_Body Body;
  slBus43_Wheels Wheels;
  SteeringBus Steering;
  SuspensionBus Suspension;
  TrlrBodyBus TrlrBody;
  TrlrWheelsBus TrlrWheels;
  CntrlFdbkBus CntrlFdbk;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_Contact_
#define DEFINED_TYPEDEF_FOR_Contact_

enum class Contact
  : int8_T {
  Init = 0,
  Open,                                /* Default value */
  Close
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_MonitorCellTempModeType_
#define DEFINED_TYPEDEF_FOR_MonitorCellTempModeType_

enum class MonitorCellTempModeType
  : int32_T {
  None = 0,                            /* Default value */
  NoCellTempFault,
  HighTemperatureFault,
  LowTemperatureFault
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_MonitorCellVoltageModeType_
#define DEFINED_TYPEDEF_FOR_MonitorCellVoltageModeType_

enum class MonitorCellVoltageModeType
  : int32_T {
  None = 0,                            /* Default value */
  NoCellVoltFault,
  OverVoltageFault,
  SensorFaut,
  UnderVoltageFault
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_MonitorCurrLimModeType_
#define DEFINED_TYPEDEF_FOR_MonitorCurrLimModeType_

enum class MonitorCurrLimModeType
  : int32_T {
  None = 0,                            /* Default value */
  NoCurrLimFault,
  Wait,
  OverCurrentFault
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_struct_beekyrb3065f5ZXL7GuueB_
#define DEFINED_TYPEDEF_FOR_struct_beekyrb3065f5ZXL7GuueB_

struct struct_beekyrb3065f5ZXL7GuueB
{
  real_T CellVoltages[96];
  real_T PackVoltage;
  real_T PackCurrent;
  real_T CellTemperatures[96];
  real_T VoutChgr;
  real_T VoutInvtr;
};

#endif

/* Forward declaration for rtModel */
typedef struct tag_RTM_CavLyriqModel_T RT_MODEL_CavLyriqModel_T;

#endif                                 /* RTW_HEADER_CavLyriqModel_types_h_ */
