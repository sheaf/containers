{-# LANGUAGE CPP #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE PatternGuards #-}
#ifdef __GLASGOW_HASKELL__
{-# LANGUAGE DeriveLift #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
#endif
#if !defined(TESTING) && defined(__GLASGOW_HASKELL__)
{-# LANGUAGE Trustworthy #-}
#endif

{-# OPTIONS_HADDOCK not-home #-}
{-# OPTIONS_GHC -fno-warn-incomplete-uni-patterns #-}

#include "containers.h"

-----------------------------------------------------------------------------
-- |
-- Module      :  Data.IntMap.NonEmpty.Internal
-- License     :  BSD-style
-- Maintainer  :  libraries@haskell.org
-- Portability :  portable
--
-- = WARNING
--
-- This module is considered __internal__.
--
-- The Package Versioning Policy __does not apply__.
--
-- The contents of this module may change __in any way whatsoever__
-- and __without any warning__ between minor versions of this package.
--
-- Authors importing this module are expected to track development
-- closely.
--
-- = Description
--
-- This defines the data structures and core (hidden) manipulations
-- on representations.
--

module Data.IntMap.NonEmpty.Strict.Internal (
    -- * Map type
    IntMap(..), Key          -- instance Eq,Show

    -- * Construction
    , singleton
    , fromSet

    -- ** From Unordered Lists
    , fromList
    , fromListWith
    , fromListWithKey

    -- ** From Ascending Lists
    , fromAscList
    , fromAscListWith
    , fromAscListWithKey
    , fromDistinctAscList

    -- ** Internal helpers
    , fromMonoListWithKey
    , Distinct(..)

    -- * Insertion
    , insert
    , insertWith
    , insertWithKey
    , insertLookupWithKey

    -- * Update
    , adjust
    , adjustWithKey

    -- * Query
    -- ** Lookup
    , lookup
    , (!?)
    , (!)
    , findWithDefault
    , member
    , notMember
    , lookupLT
    , lookupGT
    , lookupLE
    , lookupGE

    -- ** Size
    , size

    -- * Combine

    -- ** Union
    , union
    , unionWith
    , unionWithKey

    -- ** Disjoint
    , disjoint

    -- ** Universal combining function
    , mergeWithKey

    -- * Traversal
    -- ** Map
    , map
    , mapWithKey
    , traverseWithKey
    , mapAccum
    , mapAccumL
    , mapAccumWithKey
    , mapAccumRWithKey
    , mapKeys
    , mapKeysWith
    , mapKeysMonotonic

    -- * Folds
    , foldr
    , foldl
    , foldrWithKey
    , foldlWithKey
    , foldMapWithKey

    -- ** Strict folds
    , foldr'
    , foldl'
    , foldrWithKey'
    , foldlWithKey'

    -- * Conversion
    , elems
    , keys
    , assocs
    , keysSet

    -- ** Lists
    , toList

-- ** Ordered lists
    , toAscList
    , toDescList

    -- * Splitting
    , splitRoot

    -- * Submap
    , isSubmapOf, isSubmapOfBy
    , isProperSubmapOf, isProperSubmapOfBy

    -- * Min\/Max
    , findMin
    , findMax
    , updateMinWithKey
    , updateMaxWithKey

    ) where

#if !(MIN_VERSION_base(4,11,0))
import Data.Semigroup (Semigroup((<>)))
#endif

import Data.Bits
import qualified Data.IntMap.NonEmpty.Internal as L
import Data.IntMap.NonEmpty.Internal
  ( IntMap (..)
  , Key
  , mask
  , branchMask
  , nomatch
  , zero
  , natFromInt
  , intFromNat
  , link
  , linkWithMask

  , (!)
  , (!?)
  , assocs
  , findMin
  , findMax
  , findWithDefault
  , foldMapWithKey
  , foldr
  , foldl
  , foldr'
  , foldl'
  , foldlWithKey
  , foldrWithKey
  , foldlWithKey'
  , foldrWithKey'
  , keysSet
  , elems
  , disjoint
  , isProperSubmapOf
  , isProperSubmapOfBy
  , isSubmapOf
  , isSubmapOfBy
  , lookup
  , lookupLE
  , lookupGE
  , lookupLT
  , lookupGT
  , keys
  , member
  , notMember
  , size
  , splitRoot
  , toAscList
  , toDescList
  , toList
  , union
  )
import qualified Data.Foldable as Foldable
import Utils.Containers.Internal.Prelude hiding
  (lookup, map, filter, foldr, foldl, null)
import Prelude ()


import qualified Data.IntSet.Internal as IntSet
import Utils.Containers.Internal.BitUtil
import Utils.Containers.Internal.StrictPair

#ifdef __GLASGOW_HASKELL__
import Data.Coerce
#endif


{--------------------------------------------------------------------
  Construction
--------------------------------------------------------------------}

-- | \(O(1)\). A map of one element.
--
-- > singleton 1 'a'        == fromList [(1, 'a')]
-- > size (singleton 1 'a') == 1

singleton :: Key -> a -> IntMap a
singleton k !x = L.singleton k x
{-# INLINE singleton #-}

{--------------------------------------------------------------------
  Insert
--------------------------------------------------------------------}
-- | \(O(\min(n,W))\). Insert a new key\/value pair in the map.
-- If the key is already present in the map, the associated value is
-- replaced with the supplied value, i.e. 'insert' is equivalent to
-- @'insertWith' 'const'@.
--
-- > insert 5 'x' (fromList [(5,'a'), (3,'b')]) == fromList [(3, 'b'), (5, 'x')]
-- > insert 7 'x' (fromList [(5,'a'), (3,'b')]) == fromList [(3, 'b'), (5, 'a'), (7, 'x')]
-- > insert 5 'x' empty                         == singleton 5 'x'

insert :: Key -> a -> IntMap a -> IntMap a
insert !k !x t = L.insert k x t

-- right-biased insertion, used by 'union'
-- | \(O(\min(n,W))\). Insert with a combining function.
-- @'insertWith' f key value mp@
-- will insert the pair (key, value) into @mp@ if key does
-- not exist in the map. If the key does exist, the function will
-- insert @f new_value old_value@.
--
-- > insertWith (++) 5 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "xxxa")]
-- > insertWith (++) 7 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a"), (7, "xxx")]
-- > insertWith (++) 5 "xxx" empty                         == singleton 5 "xxx"

insertWith :: (a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWith f k x t
  = insertWithKey (\_ x' y' -> f x' y') k x t

-- | \(O(\min(n,W))\). Insert with a combining function.
-- @'insertWithKey' f key value mp@
-- will insert the pair (key, value) into @mp@ if key does
-- not exist in the map. If the key does exist, the function will
-- insert @f key new_value old_value@.
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > insertWithKey f 5 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "5:xxx|a")]
-- > insertWithKey f 7 "xxx" (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a"), (7, "xxx")]
-- > insertWithKey f 5 "xxx" empty                         == singleton 5 "xxx"

insertWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> IntMap a
insertWithKey f !k x t =
  case t of
    Bin p m l r
      | nomatch k p m -> link k (singleton k x) p t
      | zero k m      -> Bin p m (insertWithKey f k x l) r
      | otherwise     -> Bin p m l (insertWithKey f k x r)
    Tip ky y
      | k==ky         -> Tip k $! f k x y
      | otherwise     -> link k (singleton k x) ky t

-- | \(O(\min(n,W))\). The expression (@'insertLookupWithKey' f k x map@)
-- is a pair where the first element is equal to (@'lookup' k map@)
-- and the second element equal to (@'insertWithKey' f k x map@).
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > insertLookupWithKey f 5 "xxx" (fromList [(5,"a"), (3,"b")]) == (Just "a", fromList [(3, "b"), (5, "5:xxx|a")])
-- > insertLookupWithKey f 7 "xxx" (fromList [(5,"a"), (3,"b")]) == (Nothing,  fromList [(3, "b"), (5, "a"), (7, "xxx")])
-- > insertLookupWithKey f 5 "xxx" empty                         == (Nothing,  singleton 5 "xxx")
--
-- This is how to define @insertLookup@ using @insertLookupWithKey@:
--
-- > let insertLookup kx x t = insertLookupWithKey (\_ a _ -> a) kx x t
-- > insertLookup 5 "x" (fromList [(5,"a"), (3,"b")]) == (Just "a", fromList [(3, "b"), (5, "x")])
-- > insertLookup 7 "x" (fromList [(5,"a"), (3,"b")]) == (Nothing,  fromList [(3, "b"), (5, "a"), (7, "x")])

insertLookupWithKey :: (Key -> a -> a -> a) -> Key -> a -> IntMap a -> (Maybe a, IntMap a)
insertLookupWithKey f0 !k0 x0 t0 = toPair $ go f0 k0 x0 t0
  where
    go f k x t =
      case t of
        Bin p m l r
          | nomatch k p m -> Nothing :*: link k (singleton k x) p t
          | zero k m      -> let (found :*: l') = go f k x l in (found :*: Bin p m l' r)
          | otherwise     -> let (found :*: r') = go f k x r in (found :*: Bin p m l r')
        Tip ky y
          | k==ky         -> (Just y :*: (Tip k $! f k x y))
          | otherwise     -> (Nothing :*: link k (singleton k x) ky t)

{--------------------------------------------------------------------
  Update
--------------------------------------------------------------------}

-- | \(O(\min(n,W))\). Adjust a value at a specific key. When the key is not
-- a member of the map, the original map is returned.
--
-- > adjust ("new " ++) 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "new a")]
-- > adjust ("new " ++) 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > adjust ("new " ++) 7 empty                         == empty

adjust ::  (a -> a) -> Key -> IntMap a -> IntMap a
adjust f k m
  = adjustWithKey (\_ x -> f x) k m

-- | \(O(\min(n,W))\). Adjust a value at a specific key. When the key is not
-- a member of the map, the original map is returned.
--
-- > let f key x = (show key) ++ ":new " ++ x
-- > adjustWithKey f 5 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "5:new a")]
-- > adjustWithKey f 7 (fromList [(5,"a"), (3,"b")]) == fromList [(3, "b"), (5, "a")]
-- > adjustWithKey f 7 empty                         == empty

adjustWithKey ::  (Key -> a -> a) -> Key -> IntMap a -> IntMap a
adjustWithKey f !k (Bin p m l r)
  | zero k m      = Bin p m (adjustWithKey f k l) r
  | otherwise     = Bin p m l (adjustWithKey f k r)
adjustWithKey f k t@(Tip ky y)
  | k == ky       = Tip ky $! f k y
  | otherwise     = t

{--------------------------------------------------------------------
  Union
--------------------------------------------------------------------}

-- | \(O(n+m)\). The union with a combining function.
--
-- > unionWith (++) (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "aA"), (7, "C")]

unionWith :: (a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWith f m1 m2
  = unionWithKey (\_ x y -> f x y) m1 m2

-- | \(O(n+m)\). The union with a combining function.
--
-- > let f key left_value right_value = (show key) ++ ":" ++ left_value ++ "|" ++ right_value
-- > unionWithKey f (fromList [(5, "a"), (3, "b")]) (fromList [(5, "A"), (7, "C")]) == fromList [(3, "b"), (5, "5:a|A"), (7, "C")]

unionWithKey :: (Key -> a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
unionWithKey f m1 m2
  = L.mergeWithKey (\(Tip k1 x1) (Tip _k2 x2) -> Tip k1 $! f k1 x1 x2)
      id id m1 m2

{--------------------------------------------------------------------
  MergeWithKey
--------------------------------------------------------------------}

-- | \(O(n+m)\). A high-performance universal combining function. Using
-- 'mergeWithKey', all combining functions can be defined without any loss of
-- efficiency.
--
-- Please make sure you know what is going on when using 'mergeWithKey',
-- otherwise you can be surprised by unexpected code growth or even
-- corruption of the data structure.
--
-- When 'mergeWithKey' is given three arguments, it is inlined to the call
-- site. You should therefore use 'mergeWithKey' only to define your custom
-- combining functions.

mergeWithKey :: (Key -> a -> b -> c) -> (IntMap a -> IntMap c) -> (IntMap b -> IntMap c)
             -> IntMap a -> IntMap b -> IntMap c
mergeWithKey f g1 g2 = L.mergeWithKey combine g1 g2
  where -- We use the lambda form to avoid non-exhaustive pattern matches warning.
        combine = \(Tip k1 x1) (Tip _k2 x2) -> Tip k1 $! f k1 x1 x2
        {-# INLINE combine #-}
{-# INLINE mergeWithKey #-}

{--------------------------------------------------------------------
  Min\/Max
--------------------------------------------------------------------}

-- | \(O(\min(n,W))\). Update the value at the minimal key.
--
-- > updateMinWithKey (\ k a -> ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"3:b"), (5,"a")]

updateMinWithKey :: (Key -> a -> a) -> IntMap a -> IntMap a
updateMinWithKey f t =
  case t of Bin p m l r | m < 0 -> Bin p m l (go f r)
            _ -> go f t
  where
    go f' (Bin p m l r) = Bin p m (go f' l) r
    go f' (Tip k y) = Tip k $! f' k y

-- | \(O(\min(n,W))\). Update the value at the maximal key.
--
-- > updateMaxWithKey (\ k a -> ((show k) ++ ":" ++ a)) (fromList [(5,"a"), (3,"b")]) == fromList [(3,"b"), (5,"5:a")]

updateMaxWithKey :: (Key -> a -> a) -> IntMap a -> IntMap a
updateMaxWithKey f t =
  case t of Bin p m l r | m < 0 -> Bin p m (go f l) r
            _ -> go f t
  where
    go f' (Bin p m l r) = Bin p m l (go f' r)
    go f' (Tip k y) = Tip k $! f' k y

{--------------------------------------------------------------------
  Mapping
--------------------------------------------------------------------}
-- | \(O(n)\). Map a function over all values in the map.
--
-- > map (++ "x") (fromList [(5,"a"), (3,"b")]) == fromList [(3, "bx"), (5, "ax")]

map :: (a -> b) -> IntMap a -> IntMap b
map f = go
  where
    go (Bin p m l r) = Bin p m (go l) (go r)
    go (Tip k x)     = Tip k $! f x

#ifdef __GLASGOW_HASKELL__
{-# NOINLINE [1] map #-}
{-# RULES
"map/map" forall f g xs . map f (map g xs) = map (f . g) xs
"map/coerce" map coerce = coerce
 #-}
#endif

-- | \(O(n)\). Map a function over all values in the map.
--
-- > let f key x = (show key) ++ ":" ++ x
-- > mapWithKey f (fromList [(5,"a"), (3,"b")]) == fromList [(3, "3:b"), (5, "5:a")]

mapWithKey :: (Key -> a -> b) -> IntMap a -> IntMap b
mapWithKey f t
  = case t of
      Bin p m l r -> Bin p m (mapWithKey f l) (mapWithKey f r)
      Tip k x     -> Tip k $! f k x

#ifdef __GLASGOW_HASKELL__
{-# NOINLINE [1] mapWithKey #-}
{-# RULES
"mapWithKey/mapWithKey" forall f g xs . mapWithKey f (mapWithKey g xs) =
  mapWithKey (\k a -> f k (g k a)) xs
"mapWithKey/map" forall f g xs . mapWithKey f (map g xs) =
  mapWithKey (\k a -> f k (g a)) xs
"map/mapWithKey" forall f g xs . map f (mapWithKey g xs) =
  mapWithKey (\k a -> f (g k a)) xs
 #-}
#endif

-- | \(O(n)\).
-- @'traverseWithKey' f s == 'fromList' <$> 'traverse' (\(k, v) -> (,) k <$> f k v) ('toList' m)@
-- That is, behaves exactly like a regular 'traverse' except that the traversing
-- function also has access to the key associated with a value.
--
-- > traverseWithKey (\k v -> if odd k then Just (succ v) else Nothing) (fromList [(1, 'a'), (5, 'e')]) == Just (fromList [(1, 'b'), (5, 'f')])
-- > traverseWithKey (\k v -> if odd k then Just (succ v) else Nothing) (fromList [(2, 'c')])           == Nothing
traverseWithKey :: Applicative t => (Key -> a -> t b) -> IntMap a -> t (IntMap b)
traverseWithKey f = go
  where
    go (Tip k v) = (\ !v' -> Tip k v') <$> f k v
    go (Bin p m l r)
      | m < 0     = liftA2 (flip (Bin p m)) (go r) (go l)
      | otherwise = liftA2 (Bin p m) (go l) (go r)
{-# INLINE traverseWithKey #-}

-- | \(O(n)\). The function @'mapAccum'@ threads an accumulating
-- argument through the map in ascending order of keys.
--
-- > let f a b = (a ++ b, b ++ "X")
-- > mapAccum f "Everything: " (fromList [(5,"a"), (3,"b")]) == ("Everything: ba", fromList [(3, "bX"), (5, "aX")])

mapAccum :: (a -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccum f = mapAccumWithKey (\a' _ x -> f a' x)

-- | \(O(n)\). The function @'mapAccumWithKey'@ threads an accumulating
-- argument through the map in ascending order of keys.
--
-- > let f a k b = (a ++ " " ++ (show k) ++ "-" ++ b, b ++ "X")
-- > mapAccumWithKey f "Everything:" (fromList [(5,"a"), (3,"b")]) == ("Everything: 3-b 5-a", fromList [(3, "bX"), (5, "aX")])

mapAccumWithKey :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumWithKey f a t
  = mapAccumL f a t

-- | \(O(n)\). The function @'mapAccumL'@ threads an accumulating
-- argument through the map in ascending order of keys.
mapAccumL :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumL f0 a0 t0 = toPair $ go f0 a0 t0
  where
    go f a t
      = case t of
          Bin p m l r
            | m < 0 ->
                let (a1 :*: r') = go f a r
                    (a2 :*: l') = go f a1 l
                in (a2 :*: Bin p m l' r')
            | otherwise ->
                let (a1 :*: l') = go f a l
                    (a2 :*: r') = go f a1 r
                in (a2 :*: Bin p m l' r')
          Tip k x     -> let !(a',!x') = f a k x in (a' :*: Tip k x')

-- | \(O(n)\). The function @'mapAccumRWithKey'@ threads an accumulating
-- argument through the map in descending order of keys.
mapAccumRWithKey :: (a -> Key -> b -> (a,c)) -> a -> IntMap b -> (a,IntMap c)
mapAccumRWithKey f0 a0 t0 = toPair $ go f0 a0 t0
  where
    go f a t
      = case t of
          Bin p m l r
            | m < 0 ->
              let (a1 :*: l') = go f a l
                  (a2 :*: r') = go f a1 r
              in (a2 :*: Bin p m l' r')
            | otherwise ->
              let (a1 :*: r') = go f a r
                  (a2 :*: l') = go f a1 l
              in (a2 :*: Bin p m l' r')
          Tip k x     -> let !(a',!x') = f a k x in (a' :*: Tip k x')

-- | \(O(n \min(n,W))\).
-- @'mapKeys' f s@ is the map obtained by applying @f@ to each key of @s@.
--
-- The size of the result may be smaller if @f@ maps two or more distinct
-- keys to the same new key.  In this case the value at the greatest of the
-- original keys is retained.
--
-- > mapKeys (+ 1) (fromList [(5,"a"), (3,"b")])                        == fromList [(4, "b"), (6, "a")]
-- > mapKeys (\ _ -> 1) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 1 "c"
-- > mapKeys (\ _ -> 3) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 3 "c"

mapKeys :: (Key->Key) -> IntMap a -> IntMap a
mapKeys f = fromList . foldrWithKey (\k x xs -> (f k, x) : xs) []

-- | \(O(n \min(n,W))\).
-- @'mapKeysWith' c f s@ is the map obtained by applying @f@ to each key of @s@.
--
-- The size of the result may be smaller if @f@ maps two or more distinct
-- keys to the same new key.  In this case the associated values will be
-- combined using @c@.
--
-- > mapKeysWith (++) (\ _ -> 1) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 1 "cdab"
-- > mapKeysWith (++) (\ _ -> 3) (fromList [(1,"b"), (2,"a"), (3,"d"), (4,"c")]) == singleton 3 "cdab"

mapKeysWith :: (a -> a -> a) -> (Key->Key) -> IntMap a -> IntMap a
mapKeysWith c f
  = fromListWith c . foldrWithKey (\k x xs -> (f k, x) : xs) []

-- | \(O(n \min(n,W))\).
-- @'mapKeysMonotonic' f s == 'mapKeys' f s@, but works only when @f@
-- is strictly monotonic.
-- That is, for any values @x@ and @y@, if @x@ < @y@ then @f x@ < @f y@.
-- /The precondition is not checked./
-- Semi-formally, we have:
--
-- > and [x < y ==> f x < f y | x <- ls, y <- ls]
-- >                     ==> mapKeysMonotonic f s == mapKeys f s
-- >     where ls = keys s
--
-- This means that @f@ maps distinct original keys to distinct resulting keys.
-- This function has slightly better performance than 'mapKeys'.
--
-- > mapKeysMonotonic (\ k -> k * 2) (fromList [(5,"a"), (3,"b")]) == fromList [(6, "b"), (10, "a")]

mapKeysMonotonic :: (Key->Key) -> IntMap a -> IntMap a
mapKeysMonotonic f
  = fromDistinctAscList . foldrWithKey (\k x xs -> (f k, x) : xs) []

{--------------------------------------------------------------------
  Conversions
--------------------------------------------------------------------}

-- | \(O(n)\). Build a map from a set of keys and a function which for each key
-- computes its value.
--
-- > fromSet (\k -> replicate k 'a') (Data.IntSet.fromList [3, 5]) == fromList [(5,"aaaaa"), (3,"aaa")]
-- > fromSet undefined Data.IntSet.empty == empty

fromSet :: (Key -> a) -> IntSet.IntSet -> IntMap a
fromSet _ IntSet.Nil = error "fromSet: empty set"
fromSet f (IntSet.Bin p m l r) = Bin p m (fromSet f l) (fromSet f r)
fromSet f (IntSet.Tip kx bm) = buildTree f kx bm (IntSet.suffixBitMask + 1)
  where -- This is slightly complicated, as we to convert the dense
        -- representation of IntSet into tree representation of IntMap.
        --
        -- We are given a nonzero bit mask 'bmask' of 'bits' bits with prefix 'prefix'.
        -- We split bmask into halves corresponding to left and right subtree.
        -- If they are both nonempty, we create a Bin node, otherwise exactly
        -- one of them is nonempty and we construct the IntMap from that half.
        buildTree g !prefix !bmask bits = case bits of
          0 -> Tip prefix $! g prefix
          _ -> case intFromNat ((natFromInt bits) `shiftRL` 1) of
                 bits2 | bmask .&. ((1 `shiftLL` bits2) - 1) == 0 ->
                           buildTree g (prefix + bits2) (bmask `shiftRL` bits2) bits2
                       | (bmask `shiftRL` bits2) .&. ((1 `shiftLL` bits2) - 1) == 0 ->
                           buildTree g prefix bmask bits2
                       | otherwise ->
                           Bin prefix bits2 (buildTree g prefix bmask bits2) (buildTree g (prefix + bits2) (bmask `shiftRL` bits2) bits2)

{--------------------------------------------------------------------
  Lists
--------------------------------------------------------------------}

-- | \(O(n \min(n,W))\). Create a map from a list of key\/value pairs.
--
-- > fromList [] == empty
-- > fromList [(5,"a"), (3,"b"), (5, "c")] == fromList [(5,"c"), (3,"b")]
-- > fromList [(5,"c"), (3,"b"), (5, "a")] == fromList [(5,"a"), (3,"b")]

fromList :: [(Key,a)] -> IntMap a
fromList ((x0,a) : xs)
  = Foldable.foldl' ins (singleton x0 a) xs
  where
    ins t (k,x)  = insert k x t
fromList [] = error "fromList: empty list"

-- | \(O(n \min(n,W))\). Create a map from a list of key\/value pairs with a combining function. See also 'fromAscListWith'.
--
-- > fromListWith (++) [(5,"a"), (5,"b"), (3,"b"), (3,"a"), (5,"c")] == fromList [(3, "ab"), (5, "cba")]
-- > fromListWith (++) [] == empty

fromListWith :: (a -> a -> a) -> [(Key,a)] -> IntMap a
fromListWith f xs
  = fromListWithKey (\_ x y -> f x y) xs

-- | \(O(n \min(n,W))\). Build a map from a list of key\/value pairs with a combining function. See also fromAscListWithKey'.
--
-- > let f key new_value old_value = show key ++ ":" ++ new_value ++ "|" ++ old_value
-- > fromListWithKey f [(5,"a"), (5,"b"), (3,"b"), (3,"a"), (5,"c")] == fromList [(3, "3:a|b"), (5, "5:c|5:b|a")]
-- > fromListWithKey f [] == empty

fromListWithKey :: (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromListWithKey f xs
  = Foldable.foldl' ins (errorWithoutStackTrace "fromListWithKey") xs
  where
    ins t (k,x) = insertWithKey f k x t

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order.
--
-- > fromAscList [(3,"b"), (5,"a")]          == fromList [(3, "b"), (5, "a")]
-- > fromAscList [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "b")]

fromAscList :: [(Key,a)] -> IntMap a
fromAscList = fromMonoListWithKey Nondistinct (\_ x _ -> x)
{-# NOINLINE fromAscList #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order, with a combining function on equal keys.
-- /The precondition (input list is ascending) is not checked./
--
-- > fromAscListWith (++) [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "ba")]

fromAscListWith :: (a -> a -> a) -> [(Key,a)] -> IntMap a
fromAscListWith f = fromMonoListWithKey Nondistinct (\_ x y -> f x y)
{-# NOINLINE fromAscListWith #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order, with a combining function on equal keys.
-- /The precondition (input list is ascending) is not checked./
--
-- > let f key new_value old_value = (show key) ++ ":" ++ new_value ++ "|" ++ old_value
-- > fromAscListWithKey f [(3,"b"), (5,"a"), (5,"b")] == fromList [(3, "b"), (5, "5:b|a")]

fromAscListWithKey :: (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromAscListWithKey f = fromMonoListWithKey Nondistinct f
{-# NOINLINE fromAscListWithKey #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs where
-- the keys are in ascending order and all distinct.
-- /The precondition (input list is strictly ascending) is not checked./
--
-- > fromDistinctAscList [(3,"b"), (5,"a")] == fromList [(3, "b"), (5, "a")]

fromDistinctAscList :: [(Key,a)] -> IntMap a
fromDistinctAscList = fromMonoListWithKey Distinct (\_ x _ -> x)
{-# NOINLINE fromDistinctAscList #-}

-- | \(O(n)\). Build a map from a list of key\/value pairs with monotonic keys
-- and a combining function.
--
-- The precise conditions under which this function works are subtle:
-- For any branch mask, keys with the same prefix w.r.t. the branch
-- mask must occur consecutively in the list.

fromMonoListWithKey :: Distinct -> (Key -> a -> a -> a) -> [(Key,a)] -> IntMap a
fromMonoListWithKey distinct f = go
  where
    go []              = error "fromMonoListWithKey: empty list"
    go ((kx,vx) : zs1) = addAll' kx vx zs1

    -- `addAll'` collects all keys equal to `kx` into a single value,
    -- and then proceeds with `addAll`.
    addAll' !kx !vx []
        = Tip kx vx
    addAll' kx vx ((ky,vy) : zs)
        | Nondistinct <- distinct, kx == ky
        = let !v = f kx vy vx in addAll' ky v zs
        -- inlined: | otherwise = addAll kx (Tip kx vx) (ky : zs)
        | m <- branchMask kx ky
        , Inserted ty zs' <- addMany' m ky vy zs
        = addAll kx (linkWithMask m ky ty {-kx-} (Tip kx vx)) zs'

    -- for `addAll` and `addMany`, kx is /a/ key inside the tree `tx`
    -- `addAll` consumes the rest of the list, adding to the tree `tx`
    addAll _kx !tx []
        = tx
    addAll kx tx ((ky,vy) : zs)
        | m <- branchMask kx ky
        , Inserted ty zs' <- addMany' m ky vy zs
        = addAll kx (linkWithMask m ky ty {-kx-} tx) zs'

    -- `addMany'` is similar to `addAll'`, but proceeds with `addMany'`.
    addMany' _m kx vx []
        = Inserted (Tip kx vx) []
    addMany' m kx vx zs0@((ky,vy) : zs)
        | Nondistinct <- distinct, kx == ky
        = let !v = f kx vy vx in addMany' m ky v zs
        -- inlined: | otherwise = addMany m kx (Tip kx vx) (ky : zs)
        | mask kx m /= mask ky m
        = Inserted (Tip kx vx) zs0
        | mxy <- branchMask kx ky
        , Inserted ty zs' <- addMany' mxy ky vy zs
        = addMany m kx (linkWithMask mxy ky ty {-kx-} (Tip kx vx)) zs'

    -- `addAll` adds to `tx` all keys whose prefix w.r.t. `m` agrees with `kx`.
    addMany _m _kx tx []
        = Inserted tx []
    addMany m kx tx zs0@((ky,vy) : zs)
        | mask kx m /= mask ky m
        = Inserted tx zs0
        | mxy <- branchMask kx ky
        , Inserted ty zs' <- addMany' mxy ky vy zs
        = addMany m kx (linkWithMask mxy ky ty {-kx-} tx) zs'
{-# INLINE fromMonoListWithKey #-}

data Inserted a = Inserted !(IntMap a) ![(Key,a)]

data Distinct = Distinct | Nondistinct
