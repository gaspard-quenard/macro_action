(define
 (problem pfile_35_070)
 (:domain robot)
 (:objects o1
           o2
           o3
           o4
           o5
           o6
           o7
           o8
           o9
           o10
           o11
           o12
           o13
           o14
           o15
           o16
           o17
           o18
           o19
           o20
           o21
           o22
           o23
           o24
           o25
           o26
           o27
           o28
           o29
           o30
           o31
           o32
           o33
           o34
           o35
           o36
           o37
           o38
           o39
           o40
           o41
           o42
           o43
           o44
           o45
           o46
           o47
           o48
           o49
           o50
           o51
           o52
           o53
           o54
           o55
           o56
           o57
           o58
           o59
           o60
           o61
           o62
           o63
           o64
           o65
           o66
           o67
           o68
           o69
           o70
           - PACKAGE
           c
           r1
           r2
           r3
           r4
           r5
           r6
           r7
           r8
           r9
           r10
           r11
           r12
           r13
           r14
           r15
           r16
           r17
           r18
           r19
           r20
           r21
           r22
           r23
           r24
           r25
           r26
           r27
           r28
           r29
           r30
           r31
           r32
           r33
           r34
           r35
           - ROOM
           d815
           d1534
           d315
           d1523
           d2329
           d2326
           d526
           d2526
           d517
           d1629
           d016
           d033
           d2733
           d020
           d2035
           d2731
           d1927
           d1924
           d120
           d111
           d1835
           d915
           d930
           d730
           d922
           d1422
           d622
           d912
           d310
           d1028
           d913
           d14
           d621
           d211
           d2832
           - ROOMDOOR)
 (:init
  (rloc c)
  (armempty)
  (door c r16 d016)
  (door c r20 d020)
  (door c r33 d033)
  (door r1 r4 d14)
  (door r1 r11 d111)
  (door r1 r20 d120)
  (door r2 r11 d211)
  (door r3 r10 d310)
  (door r3 r15 d315)
  (door r4 r1 d14)
  (door r5 r17 d517)
  (door r5 r26 d526)
  (door r6 r21 d621)
  (door r6 r22 d622)
  (door r7 r30 d730)
  (door r8 r15 d815)
  (door r9 r12 d912)
  (door r9 r13 d913)
  (door r9 r15 d915)
  (door r9 r22 d922)
  (door r9 r30 d930)
  (door r10 r3 d310)
  (door r10 r28 d1028)
  (door r11 r1 d111)
  (door r11 r2 d211)
  (door r12 r9 d912)
  (door r13 r9 d913)
  (door r14 r22 d1422)
  (door r15 r3 d315)
  (door r15 r8 d815)
  (door r15 r9 d915)
  (door r15 r23 d1523)
  (door r15 r34 d1534)
  (door r16 c d016)
  (door r16 r29 d1629)
  (door r17 r5 d517)
  (door r18 r35 d1835)
  (door r19 r24 d1924)
  (door r19 r27 d1927)
  (door r20 c d020)
  (door r20 r1 d120)
  (door r20 r35 d2035)
  (door r21 r6 d621)
  (door r22 r6 d622)
  (door r22 r9 d922)
  (door r22 r14 d1422)
  (door r23 r15 d1523)
  (door r23 r26 d2326)
  (door r23 r29 d2329)
  (door r24 r19 d1924)
  (door r25 r26 d2526)
  (door r26 r5 d526)
  (door r26 r23 d2326)
  (door r26 r25 d2526)
  (door r27 r19 d1927)
  (door r27 r31 d2731)
  (door r27 r33 d2733)
  (door r28 r10 d1028)
  (door r28 r32 d2832)
  (door r29 r16 d1629)
  (door r29 r23 d2329)
  (door r30 r7 d730)
  (door r30 r9 d930)
  (door r31 r27 d2731)
  (door r32 r28 d2832)
  (door r33 c d033)
  (door r33 r27 d2733)
  (door r34 r15 d1534)
  (door r35 r18 d1835)
  (door r35 r20 d2035)
  (closed d1534)
  (closed d2329)
  (closed d2326)
  (closed d517)
  (closed d1629)
  (closed d016)
  (closed d033)
  (closed d1927)
  (closed d1924)
  (closed d915)
  (closed d310)
  (closed d1028)
  (closed d913)
  (closed d14)
  (closed d2832)
  (in o1 r22)
  (in o2 r31)
  (in o3 r7)
  (in o4 r8)
  (in o5 r26)
  (in o6 r3)
  (in o7 r10)
  (in o8 r8)
  (in o9 r30)
  (in o10 r19)
  (in o11 r19)
  (in o12 r12)
  (in o13 r25)
  (in o14 r29)
  (in o15 r16)
  (in o16 r2)
  (in o17 r26)
  (in o18 r11)
  (in o19 r6)
  (in o20 r26)
  (in o21 r14)
  (in o22 r29)
  (in o23 r6)
  (in o24 r12)
  (in o25 r13)
  (in o26 r3)
  (in o27 r20)
  (in o28 r10)
  (in o29 r27)
  (in o30 r29)
  (in o31 r7)
  (in o32 r22)
  (in o33 r3)
  (in o34 r20)
  (in o35 r29)
  (in o36 r10)
  (in o37 r24)
  (in o38 r26)
  (in o39 r10)
  (in o40 r21)
  (in o41 r22)
  (in o42 r32)
  (in o43 r32)
  (in o44 r24)
  (in o45 r20)
  (in o46 r18)
  (in o47 r15)
  (in o48 r25)
  (in o49 r27)
  (in o50 r15)
  (in o51 r5)
  (in o52 r28)
  (in o53 r33)
  (in o54 r35)
  (in o55 r22)
  (in o56 r15)
  (in o57 r7)
  (in o58 r23)
  (in o59 r12)
  (in o60 r34)
  (in o61 r11)
  (in o62 r12)
  (in o63 r31)
  (in o64 r3)
  (in o65 r18)
  (in o66 r19)
  (in o67 r1)
  (in o68 r22)
  (in o69 r30)
  (in o70 r28)
   (goal_in o1 r9)
           (goal_in o2 r29)
           (goal_in o3 r35)
           (goal_in o4 r14)
           (goal_in o5 r3)
           (goal_in o6 r8)
           (goal_in o7 r27)
           (goal_in o8 r31)
           (goal_in o9 r7)
           (goal_in o10 r18)
           (goal_in o11 r21)
           (goal_in o12 r8)
           (goal_in o13 r25)
           (goal_in o14 r18)
           (goal_in o15 r8)
           (goal_in o16 r1)
           (goal_in o17 r19)
           (goal_in o18 r27)
           (goal_in o19 r24)
           (goal_in o20 r3)
           (goal_in o21 r12)
           (goal_in o22 r8)
           (goal_in o23 r28)
           (goal_in o24 r10)
           (goal_in o25 r14)
           (goal_in o26 r23)
           (goal_in o27 r8)
           (goal_in o28 r29)
           (goal_in o29 r35)
           (goal_in o30 r2)
           (goal_in o31 r13)
           (goal_in o32 r6)
           (goal_in o33 r3)
           (goal_in o34 r8)
           (goal_in o35 r2)
           (goal_in o36 r20)
           (goal_in o37 r27)
           (goal_in o38 r29)
           (goal_in o39 r14)
           (goal_in o40 r13)
           (goal_in o41 r23)
           (goal_in o42 r33)
           (goal_in o43 r2)
           (goal_in o44 r17)
           (goal_in o45 r17)
           (goal_in o46 r32)
           (goal_in o47 r12)
           (goal_in o48 r31)
           (goal_in o49 r20)
           (goal_in o50 r30)
           (goal_in o51 r14)
           (goal_in o52 r34)
           (goal_in o53 r4)
           (goal_in o54 r17)
           (goal_in o55 r1)
           (goal_in o56 r34)
           (goal_in o57 r10)
           (goal_in o58 r5)
           (goal_in o59 r17)
           (goal_in o60 r15)
           (goal_in o61 r18)
           (goal_in o62 r25)
           (goal_in o63 r8)
           (goal_in o64 r6)
           (goal_in o65 r33)
           (goal_in o66 r34)
           (goal_in o67 r15)
           (goal_in o68 r29)
           (goal_in o69 r28)
           (goal_in o70 r20))
 (:goal (and
         (in o1 r9)
         (in o2 r29)
         (in o3 r35)
         (in o4 r14)
         (in o5 r3)
         (in o6 r8)
         (in o7 r27)
         (in o8 r31)
         (in o9 r7)
         (in o10 r18)
         (in o11 r21)
         (in o12 r8)
         (in o13 r25)
         (in o14 r18)
         (in o15 r8)
         (in o16 r1)
         (in o17 r19)
         (in o18 r27)
         (in o19 r24)
         (in o20 r3)
         (in o21 r12)
         (in o22 r8)
         (in o23 r28)
         (in o24 r10)
         (in o25 r14)
         (in o26 r23)
         (in o27 r8)
         (in o28 r29)
         (in o29 r35)
         (in o30 r2)
         (in o31 r13)
         (in o32 r6)
         (in o33 r3)
         (in o34 r8)
         (in o35 r2)
         (in o36 r20)
         (in o37 r27)
         (in o38 r29)
         (in o39 r14)
         (in o40 r13)
         (in o41 r23)
         (in o42 r33)
         (in o43 r2)
         (in o44 r17)
         (in o45 r17)
         (in o46 r32)
         (in o47 r12)
         (in o48 r31)
         (in o49 r20)
         (in o50 r30)
         (in o51 r14)
         (in o52 r34)
         (in o53 r4)
         (in o54 r17)
         (in o55 r1)
         (in o56 r34)
         (in o57 r10)
         (in o58 r5)
         (in o59 r17)
         (in o60 r15)
         (in o61 r18)
         (in o62 r25)
         (in o63 r8)
         (in o64 r6)
         (in o65 r33)
         (in o66 r34)
         (in o67 r15)
         (in o68 r29)
         (in o69 r28)
         (in o70 r20)))
             (:tasks (task0 (achieve-goals)))
)