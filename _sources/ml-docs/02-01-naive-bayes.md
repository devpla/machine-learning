# 02. 나이브베이즈 (Naive Bayes)

- 주어진 사전확률정보를 이용하여 사후확률을 예측
- 데이터를 나이브(단순)하게 독립적인 사건으로 가정하고 이 독립사건들을 베이즈 이론에 대입시켜 가장 높은 확률의 레이블로 분류하는 알고리즘

```
            P(A∩B)          P(B|A) * P(A)  
P(A|B) = ------------- =  -----------------
             P(B)                P(B)

P(A∩B) = P(A|B) * P(B)

            P(A∩B)    
P(B|A) = -------------
             P(A)     

P(A∩B) = P(B|A) * P(A)
```
<br/>  

## 2.1 조건부 독립

특정한 사건이 독립적 사건이라고 가정한다.  
`P(A∩B) = P(A) * P(B)`

```
               P(복권∩스팸)       P(복권|스팸) * P(스팸)  
P(스팸|복권) = --------------- =  -------------------
                 P(복권)               P(복권)
```

- `P(스팸|복권)` : 사후확률
- `P(복권|스팸)` : likelihood(가능성, 우도), 어떤 일이 있을 가능성
- `P(스팸)` : 사전확률
- `P(복권)` : 주변우도 (marginal likelihood)

|복권<br/>빈도|YES|NO|합|
|:--:|--:|--:|--:|
|스팸|3|19|22|
|햄|2|76|78|
|합|5|95|100|

```
P(스팸|복권) = P(복권|스팸) * P(스팸) / P(복권)
           = ((3/22) * (22/100)) / (5/100)
           = 0.6
          = 60%
```
결론 : 복권이라는 단어가 들어 있다면 스팸일 확률은 60%가 된다.

<table>
       <thead>
           <tr> 
               <th> </th>
               <th colspan=2>복권</th>
               <th colspan=2>돈</th>
               <th colspan=2>수신취소</th>
               <th> 합 </th>
           </tr>
           <tr>
               <td> </td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td> </td>
           </tr>
       </thead>
       <tbody>
           <tr>
               <td align = center>스팸</td>
               <td align = right>3</td>
               <td align = right>19</td>
               <td align = right>11</td>
               <td align = right>11</td>
               <td align = right>13</td>
               <td align = right>9</td>
               <td align = right>22</td>
           </tr>
           <tr>
               <td align = center>햄</td>
               <td align = right>2</td>
               <td align = right>76</td>
               <td align = right>15</td>
               <td align = right>63</td>
               <td align = right>21</td>
               <td align = right>57</td>
               <td align = right>78</td>
           </tr>
           <tr>
               <td align = center>합</td>
               <td align = right>5</td>
               <td align = right>95</td>
               <td align = right>26</td>
               <td align = right>74</td>
               <td align = right>34</td>
               <td align = right>66</td>
               <td align = right>100</td>
           </tr>
       </tbody>
</table>

```
복권 = YES, 돈 = NO, 수신취소 = YES

                                 P(복권 ∩ ㄱ돈 ∩ 수신취소 | 스팸) * P(스팸)
P(스팸 | 복권 ∩ ㄱ돈 ∩ 수신취소) =  ---------------------------------------
                                       P(복권 ∩ ㄱ돈 ∩ 수신취소)

                  P(복권 | 스팸) * P(ㄱ돈 | 스팸) * P(수신취소 | 스팸) * P(스팸)
              =  --------------------------------------------------------
                                P(복권 ∩ ㄱ돈 ∩ 수신취소)       

                ((3/22) * (11/22) * (13/22) * (22/100))
              = -----------------------------------------
                    ((5/100) * (74/100) * (34/100))
```

- ￢ 혹은 ㄱ은 not을 의미한다.

```
                                 P(복권 ∩ ㄱ돈 ∩ 수신취소 | 햄) * P(햄)
P(햄 | 복권 ∩ ㄱ돈 ∩ 수신취소) =  ---------------------------------------
                                       P(복권 ∩ ㄱ돈 ∩ 수신취소)

                  P(복권 | 햄) * P(ㄱ돈 | 햄) * P(수신취소 | 햄) * P(햄)
              =  --------------------------------------------------------
                                P(복권 ∩ ㄱ돈 ∩ 수신취소)       

                ((2/78) * (63/78) * (21/78) * (78/100))
              = -----------------------------------------
                    ((5/100) * (74/100) * (34/100))
```

```
P(스팸 | 복권 ∩ ㄱ돈 ∩ 수신취소) ∝ P(복권 ∩ ㄱ돈 ∩ 수신취소 | 스팸) * P(스팸)
P(햄 | 복권 ∩ ㄱ돈 ∩ 수신취소) ∝ P(복권 ∩ ㄱ돈 ∩ 수신취소 | 햄) * P(햄)

((3/22) * (11/22) * (13/22) * (22/100)) = 0.008863636
((2/78) * (63/78) * (21/78) * (78/100)) = 0.004349112

스팸일 확률 : 0.008863636 / (0.008863636 + 0.004349112) = 0.6708397
햄일 확률 : 0.004349112 / (0.008863636 + 0.004349112) = 0.3291603
```

<table>
       <thead>
           <tr> 
               <th> </th>
               <th colspan=2>복권</th>
               <th colspan=2>돈</th>
               <th colspan=2>수신취소</th>
               <th> 합 </th>
           </tr>
           <tr>
               <td> </td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td align = center>YES</td>
               <td align = center>NO</td>
               <td> </td>
           </tr>
       </thead>
       <tbody>
           <tr>
               <td align = center>스팸</td>
               <td align = right>3</td>
               <td align = right>19</td>
               <td align = right>22</td>
               <td align = right>0</td>
               <td align = right>13</td>
               <td align = right>9</td>
               <td align = right>22</td>
           </tr>
           <tr>
               <td align = center>햄</td>
               <td align = right>2</td>
               <td align = right>76</td>
               <td align = right>15</td>
               <td align = right>63</td>
               <td align = right>21</td>
               <td align = right>57</td>
               <td align = right>78</td>
           </tr>
           <tr>
               <td align = center>합</td>
               <td align = right>5</td>
               <td align = right>95</td>
               <td align = right>37</td>
               <td align = right>63</td>
               <td align = right>34</td>
               <td align = right>66</td>
               <td align = right>100</td>
           </tr>
       </tbody>
</table>

```
P(스팸 | 복권 ∩ ㄱ돈 ∩ 수신취소) ∝ P(복권 ∩ ㄱ돈 ∩ 수신취소 | 스팸) * P(스팸)
P(햄 | 복권 ∩ ㄱ돈 ∩ 수신취소) ∝ P(복권 ∩ ㄱ돈 ∩ 수신취소 | 햄) * P(햄)

((3/22) * (0/22) * (13/22) * (22/100)) = 0
((2/78) * (63/78) * (21/78) * (78/100)) = 0.004349112

스팸일 확률 : 0 / (0 + 0.004349112) = 0
햄일 확률 : 0.004349112 / (0 + 0.004349112) = 1
```

## 2.2 라플라스 추정량
빈도표의 각 합계에 작은 숫자를 더하는데 특징이 각 클래스에 대해 발생할 확률이 0이 되지 않도록 보장한다. 보편적으로 1로 설정한다.

```
((4/25) * (1/25) * (14/25) * (22/100)) = 0.00078848
((3/81) * (64/81) * (22/81) * (78/100)) = 0.006199597

스팸일 확률 : 0.00078848 / (0.00078848 + 0.006199597) = 0.1128322
햄일 확률 : 0.006199597 / (0.00078848 + 0.006199597) = 0.8871678
```