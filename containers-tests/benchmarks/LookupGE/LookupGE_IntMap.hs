{-# LANGUAGE CPP #-}
module LookupGE_IntMap where

import Prelude hiding (null)
import Data.IntMap.Internal
import qualified Data.IntMap.NonEmpty.Internal as NE

lookupGE1 :: Key -> IntMap a -> Maybe (Key,a)
lookupGE1 k m =
    case splitLookup k m of
        (_,Just v,_)  -> Just (k,v)
        (_,Nothing,r) -> findMinMaybe r

lookupGE2 :: Key -> IntMap a -> Maybe (Key,a)
lookupGE2 k t0 = case t0 of
    NonEmpty t
      | NE.Bin _ m l r <- t
      , m < 0
      -> if k >= 0
         then go l
         else case go r of
           Nothing -> Just $ NE.findMin l
           justx   -> justx
      | otherwise
      -> go t
    Nil -> Nothing
  where
    go (NE.Bin p m l r)
      | nomatch k p m = if k < p
        then Just $ NE.findMin l
        else Nothing
      | zero k m = case go l of
        Nothing -> Just $ NE.findMin r
        justx -> justx
      | otherwise = go r
    go (NE.Tip ky y)
      | k > ky = Nothing
      | otherwise = Just (ky, y)

lookupGE3 :: Key -> IntMap a -> Maybe (Key,a)
lookupGE3 k (NonEmpty t) = k `seq` case t of
    NE.Bin _ m l r | m < 0 ->
      if k >= 0
      then go Nothing l
      else go (Just (NE.findMin l)) r
    _ -> go Nothing t
  where
    go def (NE.Bin p m l r)
      | nomatch k p m = if k < p then Just $ NE.findMin l else def
      | zero k m  = go (Just $ NE.findMin r) l
      | otherwise = go def r
    go def (NE.Tip ky y)
      | k > ky    = def
      | otherwise = Just (ky, y)
lookupGE3 _ Nil = Nothing

lookupGE4 :: Key -> IntMap a -> Maybe (Key,a)
lookupGE4 k (NonEmpty t) = k `seq` case t of
    NE.Bin _ m l r | m < 0 -> if k >= 0 then go' l
                                        else go l r
    _ -> go' t
  where
    go def (NE.Bin p m l r)
      | nomatch k p m = if k < p then fMin l else fMin def
      | zero k m  = go r l
      | otherwise = go def r
    go def (NE.Tip ky y)
      | k > ky    = fMin def
      | otherwise = Just (ky, y)
    go' (NE.Bin p m l r)
      | nomatch k p m = if k < p then fMin l else Nothing
      | zero k m  = go r l
      | otherwise = go' r
    go' (NE.Tip ky y)
      | k > ky    = Nothing
      | otherwise = Just (ky, y)

    fMin :: NE.IntMap a -> Maybe (Key, a)
    fMin (NE.Tip ky y) = Just (ky, y)
    fMin (NE.Bin _ _ l _) = fMin l
lookupGE4 k Nil = Nothing

-------------------------------------------------------------------------------
-- Utilities
-------------------------------------------------------------------------------

-- | \(O(\log n)\). The minimal key of the map.
findMinMaybe :: IntMap a -> Maybe (Key, a)
findMinMaybe m
  | null m = Nothing
  | otherwise = Just (findMin m)

#ifdef TESTING
-------------------------------------------------------------------------------
-- Properties:
-------------------------------------------------------------------------------

prop_lookupGE12 :: Int -> [Int] -> Bool
prop_lookupGE12 x xs = case fromList $ zip xs xs of m -> lookupGE1 x m == lookupGE2 x m

prop_lookupGE13 :: Int -> [Int] -> Bool
prop_lookupGE13 x xs = case fromList $ zip xs xs of m -> lookupGE1 x m == lookupGE3 x m

prop_lookupGE14 :: Int -> [Int] -> Bool
prop_lookupGE14 x xs = case fromList $ zip xs xs of m -> lookupGE1 x m == lookupGE4 x m
#endif
