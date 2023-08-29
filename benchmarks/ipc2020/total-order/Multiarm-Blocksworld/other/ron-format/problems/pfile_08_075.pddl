(define
 (problem pfile_08_075)
 (:domain blocks)
 (:objects arm1 arm2 arm3 arm4 arm5 arm6 arm7 arm8 - ARM
           b1
           b2
           b3
           b4
           b5
           b6
           b7
           b8
           b9
           b10
           b11
           b12
           b13
           b14
           b15
           b16
           b17
           b18
           b19
           b20
           b21
           b22
           b23
           b24
           b25
           b26
           b27
           b28
           b29
           b30
           b31
           b32
           b33
           b34
           b35
           b36
           b37
           b38
           b39
           b40
           b41
           b42
           b43
           b44
           b45
           b46
           b47
           b48
           b49
           b50
           b51
           b52
           b53
           b54
           b55
           b56
           b57
           b58
           b59
           b60
           b61
           b62
           b63
           b64
           b65
           b66
           b67
           b68
           b69
           b70
           b71
           b72
           b73
           b74
           b75
           - BLOCK)
 (:init
  (hand-empty arm1)
  (hand-empty arm2)
  (hand-empty arm3)
  (hand-empty arm4)
  (hand-empty arm5)
  (hand-empty arm6)
  (hand-empty arm7)
  (hand-empty arm8)
  (clear b1)
  (on-table b73)
  (on b1 b73)
  (clear b15)
  (on-table b72)
  (on b15 b57)
  (on b57 b66)
  (on b66 b20)
  (on b20 b2)
  (on b2 b54)
  (on b54 b35)
  (on b35 b61)
  (on b61 b58)
  (on b58 b31)
  (on b31 b50)
  (on b50 b13)
  (on b13 b26)
  (on b26 b51)
  (on b51 b38)
  (on b38 b22)
  (on b22 b11)
  (on b11 b37)
  (on b37 b72)
  (clear b10)
  (on-table b45)
  (on b10 b23)
  (on b23 b45)
  (clear b44)
  (on-table b44)
  (clear b25)
  (on-table b40)
  (on b25 b56)
  (on b56 b9)
  (on b9 b70)
  (on b70 b49)
  (on b49 b34)
  (on b34 b42)
  (on b42 b47)
  (on b47 b64)
  (on b64 b19)
  (on b19 b12)
  (on b12 b40)
  (clear b52)
  (on-table b29)
  (on b52 b21)
  (on b21 b65)
  (on b65 b28)
  (on b28 b5)
  (on b5 b43)
  (on b43 b68)
  (on b68 b6)
  (on b6 b74)
  (on b74 b69)
  (on b69 b46)
  (on b46 b16)
  (on b16 b60)
  (on b60 b18)
  (on b18 b67)
  (on b67 b53)
  (on b53 b32)
  (on b32 b8)
  (on b8 b14)
  (on b14 b36)
  (on b36 b71)
  (on b71 b48)
  (on b48 b33)
  (on b33 b29)
  (clear b62)
  (on-table b27)
  (on b62 b30)
  (on b30 b3)
  (on b3 b55)
  (on b55 b41)
  (on b41 b75)
  (on b75 b59)
  (on b59 b27)
  (clear b39)
  (on-table b4)
  (on b39 b24)
  (on b24 b7)
  (on b7 b63)
  (on b63 b17)
  (on b17 b4))
 (:goal (and
         (clear b49)
         (on-table b71)
         (on b49 b20)
         (on b20 b35)
         (on b35 b71)
         (clear b40)
         (on-table b68)
         (on b40 b5)
         (on b5 b44)
         (on b44 b60)
         (on b60 b38)
         (on b38 b32)
         (on b32 b11)
         (on b11 b14)
         (on b14 b34)
         (on b34 b6)
         (on b6 b24)
         (on b24 b68)
         (clear b33)
         (on-table b65)
         (on b33 b21)
         (on b21 b22)
         (on b22 b4)
         (on b4 b25)
         (on b25 b10)
         (on b10 b18)
         (on b18 b65)
         (clear b8)
         (on-table b64)
         (on b8 b45)
         (on b45 b56)
         (on b56 b63)
         (on b63 b43)
         (on b43 b64)
         (clear b39)
         (on-table b54)
         (on b39 b74)
         (on b74 b47)
         (on b47 b28)
         (on b28 b36)
         (on b36 b51)
         (on b51 b72)
         (on b72 b62)
         (on b62 b29)
         (on b29 b9)
         (on b9 b15)
         (on b15 b52)
         (on b52 b54)
         (clear b48)
         (on-table b31)
         (on b48 b67)
         (on b67 b69)
         (on b69 b58)
         (on b58 b53)
         (on b53 b1)
         (on b1 b42)
         (on b42 b75)
         (on b75 b70)
         (on b70 b26)
         (on b26 b41)
         (on b41 b59)
         (on b59 b66)
         (on b66 b31)
         (clear b23)
         (on-table b19)
         (on b23 b37)
         (on b37 b61)
         (on b61 b73)
         (on b73 b57)
         (on b57 b55)
         (on b55 b30)
         (on b30 b13)
         (on b13 b27)
         (on b27 b50)
         (on b50 b12)
         (on b12 b46)
         (on b46 b7)
         (on b7 b16)
         (on b16 b17)
         (on b17 b3)
         (on b3 b2)
         (on b2 b19)))
                                      (:tasks (task0 (achieve-goals arm1)))
                                      (:tasks (task1 (achieve-goals arm2)))
                                      (:tasks (task2 (achieve-goals arm3)))
                                      (:tasks (task3 (achieve-goals arm4)))
                                      (:tasks (task4 (achieve-goals arm5)))
                                      (:tasks (task5 (achieve-goals arm6)))
                                      (:tasks (task6 (achieve-goals arm7)))
                                      (:tasks (task7 (achieve-goals arm8)))
)