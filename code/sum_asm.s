# sum_asm_kernel(a = rdi, m = rsi) -> xmm0; m % 128 == 0, m >= 128
# 16 ymm accumulators, nothing spills: 128 floats per iteration
        .text
        .globl  sum_asm_kernel
        .type   sum_asm_kernel, @function
sum_asm_kernel:
        vxorps  %ymm0, %ymm0, %ymm0
        vxorps  %ymm1, %ymm1, %ymm1
        vxorps  %ymm2, %ymm2, %ymm2
        vxorps  %ymm3, %ymm3, %ymm3
        vxorps  %ymm4, %ymm4, %ymm4
        vxorps  %ymm5, %ymm5, %ymm5
        vxorps  %ymm6, %ymm6, %ymm6
        vxorps  %ymm7, %ymm7, %ymm7
        vxorps  %ymm8, %ymm8, %ymm8
        vxorps  %ymm9, %ymm9, %ymm9
        vxorps  %ymm10, %ymm10, %ymm10
        vxorps  %ymm11, %ymm11, %ymm11
        vxorps  %ymm12, %ymm12, %ymm12
        vxorps  %ymm13, %ymm13, %ymm13
        vxorps  %ymm14, %ymm14, %ymm14
        vxorps  %ymm15, %ymm15, %ymm15
        xor     %rcx, %rcx
1:
        vaddps    0(%rdi,%rcx,4), %ymm0, %ymm0
        vaddps   32(%rdi,%rcx,4), %ymm1, %ymm1
        vaddps   64(%rdi,%rcx,4), %ymm2, %ymm2
        vaddps   96(%rdi,%rcx,4), %ymm3, %ymm3
        vaddps  128(%rdi,%rcx,4), %ymm4, %ymm4
        vaddps  160(%rdi,%rcx,4), %ymm5, %ymm5
        vaddps  192(%rdi,%rcx,4), %ymm6, %ymm6
        vaddps  224(%rdi,%rcx,4), %ymm7, %ymm7
        vaddps  256(%rdi,%rcx,4), %ymm8, %ymm8
        vaddps  288(%rdi,%rcx,4), %ymm9, %ymm9
        vaddps  320(%rdi,%rcx,4), %ymm10, %ymm10
        vaddps  352(%rdi,%rcx,4), %ymm11, %ymm11
        vaddps  384(%rdi,%rcx,4), %ymm12, %ymm12
        vaddps  416(%rdi,%rcx,4), %ymm13, %ymm13
        vaddps  448(%rdi,%rcx,4), %ymm14, %ymm14
        vaddps  480(%rdi,%rcx,4), %ymm15, %ymm15
        add     $128, %rcx
        cmp     %rsi, %rcx
        jl      1b
# 16 chains -> 1 vector -> 1 lane
        vaddps  %ymm8, %ymm0, %ymm0
        vaddps  %ymm9, %ymm1, %ymm1
        vaddps  %ymm10, %ymm2, %ymm2
        vaddps  %ymm11, %ymm3, %ymm3
        vaddps  %ymm12, %ymm4, %ymm4
        vaddps  %ymm13, %ymm5, %ymm5
        vaddps  %ymm14, %ymm6, %ymm6
        vaddps  %ymm15, %ymm7, %ymm7
        vaddps  %ymm4, %ymm0, %ymm0
        vaddps  %ymm5, %ymm1, %ymm1
        vaddps  %ymm6, %ymm2, %ymm2
        vaddps  %ymm7, %ymm3, %ymm3
        vaddps  %ymm2, %ymm0, %ymm0
        vaddps  %ymm3, %ymm1, %ymm1
        vaddps  %ymm1, %ymm0, %ymm0
        vextractf128 $1, %ymm0, %xmm1
        vaddps  %xmm1, %xmm0, %xmm0
        vpermilps $0x4e, %xmm0, %xmm1
        vaddps  %xmm1, %xmm0, %xmm0
        vpermilps $0xb1, %xmm0, %xmm1
        vaddss  %xmm1, %xmm0, %xmm0
        vzeroupper
        ret
        .size   sum_asm_kernel, .-sum_asm_kernel
