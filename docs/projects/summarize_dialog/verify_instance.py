def verify_m5_2xlarge():
    import subprocess, psutil
    c,m=int(subprocess.getoutput("nproc")),psutil.virtual_memory().total/1024**3
    assert c==8 and abs(m-32)<2,f"ERROR: Wrong instance type. Select ml.m5.2xlarge (8 vCPUs, 32 GiB). Current: {c} vCPUs, {m:.1f} GiB\nFix: File->Shut Down, then select ml.m5.2xlarge"
    print("Instance verified ✓")

verify_m5_2xlarge()